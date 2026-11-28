//! Only cache data in directories that are private to the current Windows user.
use std::{
    fs::{File, OpenOptions},
    io,
    mem::size_of,
    os::windows::{
        fs::OpenOptionsExt,
        io::{AsRawHandle, FromRawHandle, OwnedHandle},
    },
    path::Path,
    ptr,
    sync::OnceLock,
};
use windows_sys::{
    Win32::{
        Foundation::{GENERIC_ALL, GENERIC_WRITE, LocalFree},
        Security::{
            Authorization::{GetSecurityInfo, SE_FILE_OBJECT},
            *,
        },
        Storage::FileSystem::{
            DELETE, FILE_DELETE_CHILD, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
            FILE_READ_ATTRIBUTES, FILE_WRITE_ATTRIBUTES, GetFileInformationByHandle, READ_CONTROL,
            WRITE_DAC, WRITE_OWNER,
        },
        System::{
            SystemServices::{ACCESS_ALLOWED_ACE_TYPE, ACCESS_DENIED_ACE_TYPE},
            Threading::{GetCurrentProcess, OpenProcessToken},
        },
    },
    core::w,
};

fn denied() -> io::Error {
    io::ErrorKind::PermissionDenied.into()
}

// TOKEN_USER and its SID live together in the token-information buffer.
fn token_user() -> Option<Vec<usize>> {
    unsafe {
        let mut raw = ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut raw) == 0 {
            return None;
        }
        let token = OwnedHandle::from_raw_handle(raw);
        let mut needed = 0;
        GetTokenInformation(
            token.as_raw_handle(),
            TokenUser,
            ptr::null_mut(),
            0,
            &mut needed,
        );
        if (needed as usize) < size_of::<TOKEN_USER>() {
            return None;
        }
        let mut data = vec![0usize; (needed as usize).div_ceil(size_of::<usize>())];
        (GetTokenInformation(
            token.as_raw_handle(),
            TokenUser,
            data.as_mut_ptr().cast(),
            needed,
            &mut needed,
        ) != 0)
            .then_some(data)
    }
}

fn installer_sid() -> Option<Vec<u32>> {
    // The root of a system volume may be owned by the local servicing account.
    let mut sid_len = SECURITY_MAX_SID_SIZE;
    let mut sid = vec![0u32; (sid_len as usize).div_ceil(4)];
    let mut domain = [0u16; 256];
    let mut domain_len = domain.len() as u32;
    let mut kind = 0;
    unsafe {
        (LookupAccountNameW(
            ptr::null(),
            w!("NT SERVICE\\TrustedInstaller"),
            sid.as_mut_ptr().cast(),
            &mut sid_len,
            domain.as_mut_ptr(),
            &mut domain_len,
            &mut kind,
        ) != 0)
            .then_some(sid)
    }
}

// sid points into a security descriptor supplied and validated by Windows.
fn trusted(sid: PSID) -> bool {
    static USER: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    static INSTALLER: OnceLock<Option<Vec<u32>>> = OnceLock::new();
    unsafe {
        !sid.is_null()
            && IsValidSid(sid) != 0
            && (IsWellKnownSid(sid, WinLocalSystemSid) != 0
                || IsWellKnownSid(sid, WinBuiltinAdministratorsSid) != 0
                || USER.get_or_init(token_user).as_ref().is_some_and(|buf| {
                    EqualSid(sid, (*buf.as_ptr().cast::<TOKEN_USER>()).User.Sid) != 0
                })
                || INSTALLER
                    .get_or_init(installer_sid)
                    .as_ref()
                    .is_some_and(|buf| EqualSid(sid, buf.as_ptr().cast_mut().cast()) != 0))
    }
}

fn check_acl(owner: PSID, acl: *const ACL, ancestor: bool) -> io::Result<()> {
    if !trusted(owner) || acl.is_null() {
        return Err(denied());
    }
    // Ancestors can be traversable, but another user must not be able to replace an
    // existing component or take control of it. Data directories also protect inherited grants.
    const CAN_REPLACE: u32 = DELETE
        | FILE_DELETE_CHILD
        | FILE_WRITE_ATTRIBUTES
        | WRITE_DAC
        | WRITE_OWNER
        | GENERIC_ALL
        | GENERIC_WRITE;
    unsafe {
        if IsValidAcl(acl) == 0 {
            return Err(denied());
        }
        for index in 0..u32::from((*acl).AceCount) {
            let mut ace = ptr::null_mut();
            if GetAce(acl, index, &mut ace) == 0 {
                return Err(io::Error::last_os_error());
            }
            let header = &*ace.cast::<ACE_HEADER>();
            let inherit_only = u32::from(header.AceFlags) & INHERIT_ONLY_ACE != 0;
            if u32::from(header.AceType) == ACCESS_DENIED_ACE_TYPE || (ancestor && inherit_only) {
                continue;
            }
            if u32::from(header.AceType) != ACCESS_ALLOWED_ACE_TYPE {
                return Err(denied());
            }
            let allow = &*ace.cast::<ACCESS_ALLOWED_ACE>();
            let trustee = ptr::addr_of!(allow.SidStart).cast_mut().cast();
            if (ancestor && allow.Mask & CAN_REPLACE == 0)
                || trusted(trustee)
                || IsWellKnownSid(trustee, WinCreatorOwnerRightsSid) != 0
                || (inherit_only && IsWellKnownSid(trustee, WinCreatorOwnerSid) != 0)
            {
                continue;
            }
            return Err(denied());
        }
    }
    Ok(())
}

pub(super) fn check_handle(file: &File, ancestor: bool) -> io::Result<()> {
    let meta = file.metadata()?;
    if super::is_link(&meta) {
        return Err(denied());
    }
    unsafe {
        let handle = file.as_raw_handle();
        if meta.is_file() {
            let mut info = std::mem::zeroed();
            if GetFileInformationByHandle(handle, &mut info) == 0 {
                return Err(io::Error::last_os_error());
            }
            if info.nNumberOfLinks != 1 {
                return Err(denied());
            }
        }
        let (mut owner, mut acl, mut descriptor) =
            (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
        let error = GetSecurityInfo(
            handle,
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            ptr::null_mut(),
            &mut acl,
            ptr::null_mut(),
            &mut descriptor,
        );
        if error != 0 {
            return Err(io::Error::from_raw_os_error(error as i32));
        }
        let result = check_acl(owner, acl, ancestor);
        LocalFree(descriptor);
        result
    }
}

pub(super) fn check_path(path: &Path, ancestor: bool) -> io::Result<()> {
    let handle = OpenOptions::new()
        .access_mode(READ_CONTROL | FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    check_handle(&handle, ancestor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::Security::Authorization::{
        ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
    };

    fn allowed(sddl: &str, ancestor: bool) -> bool {
        let text: Vec<_> = sddl.encode_utf16().chain(Some(0)).collect();
        // Windows parses and validates the descriptor and owns its allocation until LocalFree.
        unsafe {
            let mut desc = ptr::null_mut();
            assert_ne!(
                ConvertStringSecurityDescriptorToSecurityDescriptorW(
                    text.as_ptr(),
                    SDDL_REVISION_1,
                    &mut desc,
                    ptr::null_mut()
                ),
                0
            );
            let (mut owner, mut acl) = (ptr::null_mut(), ptr::null_mut());
            let (mut present, mut defaulted) = (0, 0);
            assert_ne!(
                GetSecurityDescriptorOwner(desc, &mut owner, &mut defaulted),
                0
            );
            assert_ne!(
                GetSecurityDescriptorDacl(desc, &mut present, &mut acl, &mut defaulted),
                0
            );
            let valid = check_acl(owner, acl, ancestor).is_ok();
            LocalFree(desc);
            valid
        }
    }

    #[test]
    fn windows_cache_requires_private_data_and_trusted_ancestors() {
        assert!(allowed("O:SYD:(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", false));
        assert!(!allowed("O:SYD:(A;;FR;;;WD)", false));
        assert!(!allowed("O:SYD:(A;OICIIO;FR;;;WD)", false));
        assert!(allowed("O:SYD:(A;;FR;;;WD)", true));
        assert!(!allowed("O:SYD:(A;;0x00000040;;;WD)", true));
    }
}
