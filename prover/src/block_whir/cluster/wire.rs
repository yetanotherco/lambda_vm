use std::{
    fs::File,
    io::{self, Read, Write},
    net::{Shutdown, TcpStream},
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
const VERSION: u16 = 1;
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

pub(super) fn write_frame(
    stream: &TcpStream,
    deadline: Instant,
    kind: Kind,
    attempt: Attempt,
    payload: &[u8],
) -> io::Result<()> {
    let head = header(kind, attempt, payload.len())?;
    for mut bytes in [&head[..], payload] {
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

pub(super) struct Reader<'a> {
    stream: &'a TcpStream,
    stale: usize,
}

impl<'a> Reader<'a> {
    pub fn new(stream: &'a TcpStream) -> Self {
        Self { stream, stale: 0 }
    }

    pub fn receive(
        &mut self,
        deadline: Instant,
        expected: Option<Attempt>,
        kinds: &[Kind],
    ) -> io::Result<(Attempt, Kind, AlignedVec)> {
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
            if expected.is_some_and(|expected| attempt != expected) {
                self.stale = self.stale.saturating_add(1);
                // Drain stale frames in fixed-size pieces without decoding.
                let mut scratch = [0; 8192];
                for offset in (0..len).step_by(scratch.len()) {
                    let n = (len - offset).min(scratch.len());
                    read_exact(self.stream, deadline, &mut scratch[..n])?;
                }
                continue;
            }
            if !kinds.contains(&kind) && kind != Kind::Error {
                return Err(invalid(format!("unexpected {kind:?} frame")));
            }
            let mut bytes = AlignedVec::with_capacity(len);
            bytes.resize(len, 0);
            read_exact(self.stream, deadline, &mut bytes)?;
            if self.stale > 0 {
                eprintln!("CLUSTER TCP: dropped {} stale frames", self.stale);
                self.stale = 0;
            }
            if kind == Kind::Error {
                return Err(invalid(format!(
                    "peer refused attempt: {}",
                    decode::<String>(&bytes)?
                )));
            }
            return Ok((attempt, kind, bytes));
        }
    }
}

pub(super) fn send<T>(
    stream: &TcpStream,
    deadline: Instant,
    attempt: Attempt,
    kind: Kind,
    value: &T,
) -> io::Result<()>
where
    T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, ArchiveError>>,
{
    let bytes = rkyv::to_bytes::<ArchiveError>(value).map_err(|e| invalid(e.to_string()))?;
    write_frame(stream, deadline, kind, attempt, &bytes)
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

/// HMAC over the version, both nonces, and the configured connection context.
/// The role byte makes a coordinator response unusable as a worker response.
fn authenticator(
    token: &Digest32,
    context: &Digest32,
    challenge: &Digest32,
    nonce: &Digest32,
    role: u8,
) -> Digest32 {
    let mut inner = [0x36; 64];
    let mut outer = [0x5c; 64];
    for (i, b) in token.iter().enumerate() {
        inner[i] ^= b;
        outer[i] ^= b;
    }
    let mut hash = Blake3Std::new();
    hash.update(inner);
    hash.update(b"lambda-vm cluster TCP authentication");
    hash.update(VERSION.to_be_bytes());
    hash.update(context);
    hash.update(challenge);
    hash.update(nonce);
    hash.update([role]);
    Blake3Std::new()
        .chain_update(outer)
        .chain_update(hash.finalize())
        .finalize()
        .into()
}

fn check_auth(actual: &[u8], expected: Digest32) -> io::Result<()> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .fold(0, |diff, (a, b)| diff | (a ^ b))
            != 0
    {
        return Err(invalid(
            "peer authentication failed (token, build or topology mismatch)",
        ));
    }
    Ok(())
}

pub(super) fn authenticate(
    stream: &TcpStream,
    deadline: Instant,
    token: &Digest32,
    context: &Digest32,
    coordinator: bool,
) -> io::Result<()> {
    stream.set_nodelay(true)?;
    let mut reader = Reader::new(stream);
    let handshake = [0; 16];
    if coordinator {
        let challenge = random()?;
        write_frame(stream, deadline, Kind::Challenge, handshake, &challenge)?;
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
        write_frame(stream, deadline, Kind::Welcome, handshake, &welcome)?;
    } else {
        let (_, _, challenge) = reader.receive(deadline, Some(handshake), &[Kind::Challenge])?;
        let challenge = challenge[..]
            .try_into()
            .map_err(|_| invalid("invalid challenge length"))?;
        let nonce = random()?;
        let mut response = [0; 64];
        response[..32].copy_from_slice(&nonce);
        response[32..].copy_from_slice(&authenticator(token, context, &challenge, &nonce, 0));
        write_frame(stream, deadline, Kind::Authenticate, handshake, &response)?;
        let (_, _, welcome) = reader.receive(deadline, Some(handshake), &[Kind::Welcome])?;
        check_auth(
            &welcome,
            authenticator(token, context, &challenge, &nonce, 1),
        )?;
    }
    Ok(())
}

/// Close both halves on all exits, including while another thread is in a read.
pub(super) struct Connection(pub TcpStream);

impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self.0.shutdown(Shutdown::Both);
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

    #[test]
    fn cluster_wire_refuses_bad_headers_and_truncated_payloads() {
        let attempt = [7; 16];
        let valid = header(Kind::Committed, attempt, 100).unwrap();
        for (head, body, expected) in [
            (&valid[..10], &[][..], "failed to fill"),
            (&valid[..], &[0; 20][..], "unexpected end"),
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
            let (mut sender, receiver) = pair();
            sender.write_all(head).unwrap();
            sender.write_all(body).unwrap();
            sender.shutdown(Shutdown::Write).unwrap();
            let error = Reader::new(&receiver)
                .receive(deadline(), Some(attempt), &[Kind::Committed])
                .unwrap_err();
            if expected.contains("end") || expected == "failed to fill" {
                assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
            } else {
                assert!(error.to_string().contains(expected), "{error}");
            }
        }
    }

    #[test]
    fn cluster_wire_drops_stale_frames_and_checks_payloads() {
        let (sender, receiver) = pair();
        let attempt = [7; 16];
        write_frame(&sender, deadline(), Kind::Committed, [8; 16], &[1]).unwrap();
        send(
            &sender,
            deadline(),
            attempt,
            Kind::Committed,
            &vec![1u64, 2, 3],
        )
        .unwrap();
        let mut reader = Reader::new(&receiver);
        let (_, _, bytes) = reader
            .receive(deadline(), Some(attempt), &[Kind::Committed])
            .unwrap();
        assert_eq!(decode::<Vec<u64>>(&bytes).unwrap(), [1, 2, 3]);
        write_frame(&sender, deadline(), Kind::Committed, attempt, &[1]).unwrap();
        let (_, _, bytes) = reader
            .receive(deadline(), Some(attempt), &[Kind::Committed])
            .unwrap();
        assert!(decode::<Vec<u64>>(&bytes).is_err());
    }

    #[test]
    fn cluster_wire_deadline_covers_the_entire_frame() {
        let (mut sender, receiver) = pair();
        let attempt = [7; 16];
        sender
            .write_all(&header(Kind::Committed, attempt, 100).unwrap())
            .unwrap();
        let start = Instant::now();
        let error = Reader::new(&receiver)
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
