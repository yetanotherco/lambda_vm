use std::{
    fs::File,
    io::{self, Read, Write},
    net::{Shutdown, TcpStream},
    sync::Mutex,
    time::Instant,
};

use crypto::hash::blake3::chain::Blake3Std;
use digest::Digest;
use rkyv::{
    Archive, Deserialize, Serialize,
    api::high::{HighSerializer, HighValidator},
    bytecheck::CheckBytes,
    de::Pool,
    rancor::{Error as ArchiveError, Strategy},
    ser::allocator::ArenaHandle,
    util::AlignedVec,
};

pub(super) const MAX_WORKERS: usize = 8;
pub(super) const INPUT_CHUNK: usize = 1 << 20;
const HEADER: usize = 24;
const VERSION: u16 = 2;
pub(super) type Attempt = [u8; 16];
pub(super) type Digest32 = [u8; 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub(super) enum Kind {
    Challenge = 1,
    Authenticate,
    Welcome,
    Start,
    Input,
    Assign,
    Witness,
    Committed,
    CommittedAll,
    Bind,
    Result,
    OpenedAll,
    Finish,
    Error,
}

impl Kind {
    fn parse(value: u16) -> io::Result<Self> {
        use Kind::*;
        [
            Challenge,
            Authenticate,
            Welcome,
            Start,
            Input,
            Assign,
            Witness,
            Committed,
            CommittedAll,
            Bind,
            Result,
            OpenedAll,
            Finish,
            Error,
        ]
        .into_iter()
        .find(|&kind| kind as u16 == value)
        .ok_or_else(|| invalid("unknown frame kind"))
    }

    fn limit(self) -> usize {
        match self {
            Self::Challenge | Self::Authenticate | Self::Welcome => 64,
            Self::Start | Self::Error => 4096,
            Self::Witness => 32,
            Self::Committed => 256 << 10,
            Self::Input => INPUT_CHUNK + 32,
            Self::Assign | Self::Bind => 8 << 20,
            Self::Result => 16 << 20,
            Self::CommittedAll | Self::OpenedAll | Self::Finish => 0,
        }
    }
}

pub(super) fn invalid(reason: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, reason.into())
}

pub(super) fn remaining(deadline: Instant) -> io::Result<std::time::Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|left| !left.is_zero())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "phase deadline expired"))
}

pub(super) fn random<const N: usize>() -> io::Result<[u8; N]> {
    let mut bytes = [0; N];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn header(kind: Kind, attempt: Attempt, len: usize) -> io::Result<[u8; HEADER]> {
    if len > kind.limit() || (len == 0) != (kind.limit() == 0) {
        return Err(invalid(format!("{kind:?} frame length {len} is invalid")));
    }
    let mut bytes = [0; HEADER];
    bytes[..2].copy_from_slice(&VERSION.to_be_bytes());
    bytes[2..4].copy_from_slice(&(kind as u16).to_be_bytes());
    bytes[4..20].copy_from_slice(&attempt);
    bytes[20..].copy_from_slice(&(len as u32).to_be_bytes());
    Ok(bytes)
}

fn write_parts(stream: &TcpStream, deadline: Instant, parts: &[&[u8]]) -> io::Result<()> {
    for &part in parts {
        let mut bytes = part;
        while !bytes.is_empty() {
            stream.set_write_timeout(Some(remaining(deadline)?))?;
            match (&*stream).write(bytes) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => bytes = &bytes[n..],
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

fn write_handshake(
    stream: &TcpStream,
    deadline: Instant,
    kind: Kind,
    payload: &[u8],
) -> io::Result<()> {
    write_parts(
        stream,
        deadline,
        &[&header(kind, [0; 16], payload.len())?, payload],
    )
}

fn read_exact(stream: &TcpStream, deadline: Instant, mut bytes: &mut [u8]) -> io::Result<()> {
    while !bytes.is_empty() {
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        match (&*stream).read(bytes) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => bytes = &mut bytes[n..],
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

struct FrameAuth {
    key: Digest32,
    direction: u8,
    sequence: u64,
}

impl FrameAuth {
    fn mac(&self, head: &[u8; HEADER], sequence: [u8; 8]) -> Hmac {
        let mut mac = Hmac::new(&self.key);
        mac.inner.update(b"lambda-vm cluster TCP frame");
        mac.inner.update([self.direction]);
        mac.inner.update(sequence);
        mac.inner.update(head);
        mac
    }

    fn advance(&mut self) -> io::Result<()> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| invalid("frame sequence exhausted"))?;
        Ok(())
    }
}

pub(super) struct Reader<'a> {
    stream: &'a TcpStream,
    auth: Option<&'a Mutex<FrameAuth>>,
    stale: usize,
}

impl Reader<'_> {
    pub fn receive(
        &mut self,
        deadline: Instant,
        expected: Option<Attempt>,
        kinds: &[Kind],
    ) -> io::Result<(Attempt, Kind, AlignedVec)> {
        let mut auth = self.auth.map(|state| state.lock().unwrap());
        loop {
            let mut head = [0; HEADER];
            read_exact(self.stream, deadline, &mut head)?;
            if u16::from_be_bytes(head[..2].try_into().unwrap()) != VERSION {
                return Err(invalid("unsupported frame version"));
            }
            let kind = Kind::parse(u16::from_be_bytes(head[2..4].try_into().unwrap()))?;
            let attempt = head[4..20].try_into().unwrap();
            let len = u32::from_be_bytes(head[20..].try_into().unwrap()) as usize;
            header(kind, attempt, len)?;
            let stale = expected.is_some_and(|expected| attempt != expected);
            if auth.is_none() && stale {
                return Err(invalid("invalid handshake attempt ID"));
            }
            let mut mac = if let Some(state) = auth.as_deref() {
                let mut sequence = [0; 8];
                read_exact(self.stream, deadline, &mut sequence)?;
                if u64::from_be_bytes(sequence) != state.sequence {
                    return Err(invalid("unexpected frame sequence number"));
                }
                Some(state.mac(&head, sequence))
            } else {
                None
            };
            if !(stale || kinds.contains(&kind) || kind == Kind::Error && mac.is_some()) {
                return Err(invalid(format!("unexpected {kind:?} frame")));
            }
            let mut bytes = AlignedVec::new();
            if stale {
                // Authenticate stale frames without allocating or decoding their payloads.
                let mut scratch = [0; 8192];
                for offset in (0..len).step_by(scratch.len()) {
                    let n = (len - offset).min(scratch.len());
                    read_exact(self.stream, deadline, &mut scratch[..n])?;
                    mac.as_mut().unwrap().inner.update(&scratch[..n]);
                }
            } else {
                bytes.resize(len, 0);
                read_exact(self.stream, deadline, &mut bytes)?;
                if let Some(mac) = &mut mac {
                    mac.inner.update(&bytes);
                }
            }
            if let Some(mac) = mac {
                let mut tag = [0; 32];
                read_exact(self.stream, deadline, &mut tag)?;
                if !constant_time_eq(&tag, mac.finish()) {
                    return Err(invalid("invalid frame MAC"));
                }
                auth.as_deref_mut().unwrap().advance()?;
            }
            if stale {
                self.stale = self.stale.saturating_add(1);
                continue;
            }
            if self.stale > 0 {
                eprintln!("CLUSTER TCP: dropped {} stale frames", self.stale);
                self.stale = 0;
            }
            if kind == Kind::Error {
                let reason: String = decode::<String>(&bytes)?
                    .chars()
                    .filter(|c| !c.is_control())
                    .collect();
                return Err(invalid(format!("peer refused attempt: {reason}")));
            }
            return Ok((attempt, kind, bytes));
        }
    }
}

pub(super) fn decode<T>(bytes: &AlignedVec) -> io::Result<T>
where
    T: Archive,
    T::Archived: for<'a> CheckBytes<HighValidator<'a, ArchiveError>>
        + Deserialize<T, Strategy<Pool, ArchiveError>>,
{
    use rkyv::validation::{Validator, archive::ArchiveValidator, shared::SharedValidator};
    let mut context = Validator::new(
        ArchiveValidator::with_max_depth(bytes, std::num::NonZeroUsize::new(64)),
        SharedValidator::new(),
    );
    let archived: &T::Archived =
        rkyv::api::access_with_context::<_, _, ArchiveError>(bytes, &mut context)
            .map_err(|e| invalid(e.to_string()))?;
    rkyv::api::deserialize_using(archived, &mut Pool::default())
        .map_err(|e: ArchiveError| invalid(e.to_string()))
}

struct Hmac {
    inner: Blake3Std,
    outer_pad: [u8; 64],
}

impl Hmac {
    fn new(key: &Digest32) -> Self {
        let mut inner = [0x36; 64];
        let mut outer = [0x5c; 64];
        for (i, b) in key.iter().enumerate() {
            inner[i] ^= b;
            outer[i] ^= b;
        }
        Self {
            inner: Blake3Std::new().chain_update(inner),
            outer_pad: outer,
        }
    }

    fn finish(self) -> Digest32 {
        Blake3Std::new()
            .chain_update(self.outer_pad)
            .chain_update(self.inner.finalize())
            .finalize()
            .into()
    }
}

/// HMAC over the version, both nonces, and the configured connection context.
/// Purpose 0 proves the worker's identity, 1 proves the coordinator's, and 2
/// derives a session key that neither peer sends.
fn authenticator(
    token: &Digest32,
    context: &Digest32,
    challenge: &Digest32,
    nonce: &Digest32,
    purpose: u8,
) -> Digest32 {
    let mut mac = Hmac::new(token);
    mac.inner.update(b"lambda-vm cluster TCP authentication");
    mac.inner.update(VERSION.to_be_bytes());
    mac.inner.update(context);
    mac.inner.update(challenge);
    mac.inner.update(nonce);
    mac.inner.update([purpose]);
    mac.finish()
}

fn constant_time_eq(actual: &[u8], expected: Digest32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .fold(0, |diff, (a, b)| diff | (a ^ b))
            == 0
}

fn check_auth(actual: &[u8], expected: Digest32) -> io::Result<()> {
    if !constant_time_eq(actual, expected) {
        return Err(invalid(
            "peer authentication failed (token, build or topology mismatch)",
        ));
    }
    Ok(())
}

pub(super) fn authenticate(
    stream: TcpStream,
    deadline: Instant,
    token: &Digest32,
    context: &Digest32,
    coordinator: bool,
) -> io::Result<Connection> {
    stream.set_nodelay(true)?;
    let mut reader = Reader {
        stream: &stream,
        auth: None,
        stale: 0,
    };
    let handshake = [0; 16];
    let (challenge, nonce) = if coordinator {
        let challenge = random()?;
        write_handshake(&stream, deadline, Kind::Challenge, &challenge)?;
        let (_, _, response) = reader.receive(deadline, Some(handshake), &[Kind::Authenticate])?;
        if response.len() != 64 {
            return Err(invalid("invalid authentication response length"));
        }
        let nonce = response[..32].try_into().unwrap();
        check_auth(
            &response[32..],
            authenticator(token, context, &challenge, &nonce, 0),
        )?;
        let welcome = authenticator(token, context, &challenge, &nonce, 1);
        write_handshake(&stream, deadline, Kind::Welcome, &welcome)?;
        (challenge, nonce)
    } else {
        let (_, _, challenge) = reader.receive(deadline, Some(handshake), &[Kind::Challenge])?;
        let challenge = challenge[..]
            .try_into()
            .map_err(|_| invalid("invalid challenge length"))?;
        let nonce = random()?;
        let mut response = [0; 64];
        response[..32].copy_from_slice(&nonce);
        response[32..].copy_from_slice(&authenticator(token, context, &challenge, &nonce, 0));
        write_handshake(&stream, deadline, Kind::Authenticate, &response)?;
        let (_, _, welcome) = reader.receive(deadline, Some(handshake), &[Kind::Welcome])?;
        check_auth(
            &welcome,
            authenticator(token, context, &challenge, &nonce, 1),
        )?;
        (challenge, nonce)
    };
    let key = authenticator(token, context, &challenge, &nonce, 2);
    let state = |direction| {
        Mutex::new(FrameAuth {
            key,
            direction,
            sequence: 0,
        })
    };
    Ok(Connection {
        stream,
        outgoing: state(u8::from(coordinator)),
        incoming: state(u8::from(!coordinator)),
    })
}

pub(super) struct Connection {
    stream: TcpStream,
    incoming: Mutex<FrameAuth>,
    outgoing: Mutex<FrameAuth>,
}

impl Connection {
    pub fn reader(&self) -> Reader<'_> {
        Reader {
            stream: &self.stream,
            auth: Some(&self.incoming),
            stale: 0,
        }
    }

    pub fn close(&self) {
        let _ = self.stream.shutdown(Shutdown::Both);
    }

    pub fn write_frame(
        &self,
        deadline: Instant,
        kind: Kind,
        attempt: Attempt,
        payload: &[u8],
    ) -> io::Result<()> {
        let head = header(kind, attempt, payload.len())?;
        let mut state = self.outgoing.lock().unwrap();
        let sequence = state.sequence.to_be_bytes();
        let mut mac = state.mac(&head, sequence);
        mac.inner.update(payload);
        let tag = mac.finish();
        write_parts(&self.stream, deadline, &[&head, &sequence, payload, &tag])
            .inspect_err(|_| self.close())?;
        state.advance()
    }

    pub fn send<T>(
        &self,
        deadline: Instant,
        attempt: Attempt,
        kind: Kind,
        value: &T,
    ) -> io::Result<()>
    where
        T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, ArchiveError>>,
    {
        let bytes = rkyv::to_bytes::<ArchiveError>(value).map_err(|e| invalid(e.to_string()))?;
        self.write_frame(deadline, kind, attempt, &bytes)
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::time::Duration;

    fn pair() -> (TcpStream, TcpStream) {
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let sender = TcpStream::connect(server.local_addr().unwrap()).unwrap();
        let (receiver, _) = server.accept().unwrap();
        (sender, receiver)
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(2)
    }

    fn authenticated_pair() -> (Connection, Connection) {
        let (sender, receiver) = pair();
        let token = [1; 32];
        let context = [2; 32];
        std::thread::scope(|scope| {
            let sender = scope.spawn(|| authenticate(sender, deadline(), &token, &context, true));
            let receiver = authenticate(receiver, deadline(), &token, &context, false).unwrap();
            (sender.join().unwrap().unwrap(), receiver)
        })
    }

    fn captured_frame(
        sender: &Connection,
        kind: Kind,
        attempt: Attempt,
        payload: &[u8],
    ) -> Vec<u8> {
        let state = sender.outgoing.lock().unwrap();
        let head = header(kind, attempt, payload.len()).unwrap();
        let sequence = state.sequence.to_be_bytes();
        let mut mac = state.mac(&head, sequence);
        mac.inner.update(payload);
        [&head[..], &sequence, payload, &mac.finish()].concat()
    }

    #[test]
    fn cluster_wire_refuses_bad_headers_and_truncated_payloads() {
        let attempt = [7; 16];
        let valid = header(Kind::Committed, attempt, 100).unwrap();
        for (head, body, expected) in [
            (&valid[..10], &[][..], "end"),
            (&valid[..], &[0; 20][..], "end"),
            (
                &{
                    let mut h = valid;
                    h[1] ^= 1;
                    h
                }[..],
                &[][..],
                "version",
            ),
            (
                &{
                    let mut h = valid;
                    h[20..].copy_from_slice(&u32::MAX.to_be_bytes());
                    h
                }[..],
                &[][..],
                "length",
            ),
        ] {
            let (sender, receiver) = authenticated_pair();
            (&sender.stream).write_all(head).unwrap();
            (&sender.stream).write_all(body).unwrap();
            sender.stream.shutdown(Shutdown::Write).unwrap();
            let error = receiver
                .reader()
                .receive(deadline(), Some(attempt), &[Kind::Committed])
                .unwrap_err();
            if expected == "end" {
                assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
            } else {
                assert!(error.to_string().contains(expected), "{error}");
            }
        }
    }

    #[test]
    fn cluster_wire_drops_authenticated_stale_frames_and_checks_payloads() {
        let (sender, receiver) = authenticated_pair();
        let attempt = [7; 16];
        sender
            .write_frame(deadline(), Kind::Committed, [8; 16], &[1])
            .unwrap();
        sender
            .send(deadline(), attempt, Kind::Committed, &vec![1u64, 2, 3])
            .unwrap();
        let mut reader = receiver.reader();
        let (_, _, bytes) = reader
            .receive(deadline(), Some(attempt), &[Kind::Committed])
            .unwrap();
        assert_eq!(decode::<Vec<u64>>(&bytes).unwrap(), [1, 2, 3]);
        sender
            .write_frame(deadline(), Kind::Committed, attempt, &[1])
            .unwrap();
        let (_, _, bytes) = reader
            .receive(deadline(), Some(attempt), &[Kind::Committed])
            .unwrap();
        assert!(decode::<Vec<u64>>(&bytes).is_err());
    }

    #[test]
    fn cluster_wire_refuses_tampered_frames() {
        for attempt in [[7; 16], [8; 16]] {
            let (sender, receiver) = authenticated_pair();
            let mut frame = captured_frame(&sender, Kind::Result, attempt, &[1, 2, 3]);
            frame[HEADER + 8] ^= 1;
            (&sender.stream).write_all(&frame).unwrap();
            let error = receiver
                .reader()
                .receive(deadline(), Some([7; 16]), &[Kind::Result])
                .unwrap_err();
            assert!(error.to_string().contains("frame MAC"), "{error}");
        }
    }

    #[test]
    fn cluster_wire_refuses_replayed_and_out_of_order_frames() {
        let (sender, receiver) = authenticated_pair();
        let attempt = [7; 16];
        let mut reader = receiver.reader();
        let old = captured_frame(&sender, Kind::Result, attempt, &[1, 2, 3]);
        for _ in 0..2 {
            (&sender.stream).write_all(&old).unwrap();
        }
        reader
            .receive(deadline(), Some(attempt), &[Kind::Result])
            .unwrap();
        let error = reader
            .receive(deadline(), Some(attempt), &[Kind::Result])
            .unwrap_err();
        assert!(error.to_string().contains("sequence number"), "{error}");

        let (sender, receiver) = authenticated_pair();
        let next_attempt = [8; 16];
        let mut old = old;
        old[4..20].copy_from_slice(&next_attempt);
        (&sender.stream).write_all(&old).unwrap();
        let error = receiver
            .reader()
            .receive(deadline(), Some(next_attempt), &[Kind::Result])
            .unwrap_err();
        assert!(error.to_string().contains("frame MAC"), "{error}");

        let (sender, receiver) = authenticated_pair();
        sender.outgoing.lock().unwrap().advance().unwrap();
        sender
            .write_frame(deadline(), Kind::Result, attempt, &[1, 2, 3])
            .unwrap();
        let error = receiver
            .reader()
            .receive(deadline(), Some(attempt), &[Kind::Result])
            .unwrap_err();
        assert!(error.to_string().contains("sequence number"), "{error}");
    }

    #[test]
    fn cluster_wire_only_reports_authenticated_error_text_without_controls() {
        let (sender, receiver) = pair();
        let reason = rkyv::to_bytes::<ArchiveError>(&String::from("hello\x1b\nworld")).unwrap();
        write_handshake(&sender, deadline(), Kind::Error, &reason).unwrap();
        let error = authenticate(receiver, deadline(), &[0; 32], &[0; 32], false)
            .err()
            .unwrap();
        assert_eq!(error.to_string(), "unexpected Error frame");

        let (sender, receiver) = authenticated_pair();
        sender
            .write_frame(deadline(), Kind::Error, [7; 16], &reason)
            .unwrap();
        let error = receiver
            .reader()
            .receive(deadline(), Some([7; 16]), &[Kind::Result])
            .unwrap_err();
        assert_eq!(error.to_string(), "peer refused attempt: helloworld");
    }

    #[test]
    fn cluster_wire_deadline_covers_the_entire_frame() {
        let (sender, receiver) = authenticated_pair();
        let attempt = [7; 16];
        (&sender.stream)
            .write_all(&header(Kind::Committed, attempt, 100).unwrap())
            .unwrap();
        let start = Instant::now();
        let error = receiver
            .reader()
            .receive(
                start + Duration::from_millis(80),
                Some(attempt),
                &[Kind::Committed],
            )
            .unwrap_err();
        assert!(matches!(
            error.kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
        ));
        assert!(start.elapsed() < Duration::from_secs(1));
    }
}
