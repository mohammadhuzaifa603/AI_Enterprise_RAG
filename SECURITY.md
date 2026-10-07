# Security Notes

This repository is a portfolio project. It demonstrates security-conscious architecture but is not a certified production security implementation.

## Implemented safeguards

- Upload extension and PDF signature validation
- Maximum upload size and page count
- Unique content-addressed storage names
- Optional Fernet encryption at rest
- Environment-based secret configuration
- Configurable CORS
- Document ownership and read permissions
- Authorization-aware retrieval
- Audit records
- No runtime database/uploads/secrets committed to Git

## Production hardening still required

- SSO/OIDC/SAML
- KMS/HSM-backed key management
- secret manager integration
- centralized identity and RBAC administration
- malware/sandbox scanning for uploaded files
- rate limiting
- network segmentation
- database encryption and managed backups
- security monitoring and alerting
- formal threat modeling and penetration testing
