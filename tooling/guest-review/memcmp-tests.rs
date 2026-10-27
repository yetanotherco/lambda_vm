use core::arch::asm;

#[repr(align(8))]
struct Buffer([u8;1056]);

unsafe fn compare(a:*const u8,b:*const u8,n:usize)->i64 {
    let mut result=a as i64;
    unsafe { asm!("call memcmp", inlateout("a0") result, inlateout("a1") b => _, inlateout("a2") n => _, clobber_abi("C")); }
    result
}

pub fn main() {
    let mut a=Buffer([0;1056]);let mut b=Buffer([0;1056]);let mut cases=0u64;
    for n in (0..=65).chain([127,128,129,255,256,257,1023,1024]) {
        for s in 0..8 { for d in 0..8 {
            for i in 0..n {
                let value=(i*73+i/7) as u8;
                unsafe { a.0.as_mut_ptr().add(8+s+i).write_volatile(value); b.0.as_mut_ptr().add(8+d+i).write_volatile(value); }
            }
            unsafe {
                let p=a.0.as_ptr().add(8+s);let q=b.0.as_mut_ptr().add(8+d);
                assert_eq!(compare(p,q,n),0);
                if n>0 { for i in [0,n/2,n-1] {
                    let old=q.add(i).read_volatile();let new=old^0x80;
                    q.add(i).write_volatile(new);
                    let expected=if old<new {-1} else {1};
                    assert_eq!(compare(p,q,n).signum(),expected);
                    assert_eq!(compare(q,p,n).signum(),-expected);
                    q.add(i).write_volatile(old);
                }}
            }
            cases+=1;
        }}
    }
    assert_eq!(cases,4736);
    lambda_vm_syscalls::syscalls::commit(&cases.to_le_bytes());
}
