## Security

NVIDIA is dedicated to the security and trust of our software products and services, including all source code repositories managed through our organization.

If you need to report a security issue, please use the appropriate contact points outlined below. **Please do not report security vulnerabilities through GitHub.** If a potential security issue is inadvertently reported via a public issue or pull request, NVIDIA maintainers may limit public discussion and redirect the reporter to the appropriate private disclosure channels.

## Reporting Potential Security Vulnerability in an NVIDIA Product

To report a potential security vulnerability in any NVIDIA product:
- Web: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- E-Mail: psirt@nvidia.com
    - We encourage you to use the following PGP key for secure email communication: [NVIDIA public PGP Key for communication](https://www.nvidia.com/en-us/security/pgp-key)
    - Please include the following information:
   	 - Product/Driver name and version/branch that contains the vulnerability
     - Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
   	 - Instructions to reproduce the vulnerability
   	 - Proof-of-concept or exploit code
   	 - Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

While NVIDIA currently does not have a bug bounty program, we do offer acknowledgement when an externally reported security issue is addressed under our coordinated vulnerability disclosure policy. Please visit our [Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/psirt-policies/) policies page for more information.

## NVIDIA Product Security

For all security-related concerns, please visit NVIDIA's Product Security portal at https://www.nvidia.com/en-us/security

## Loading Model Files and Artifacts

Kumo-TS loads model checkpoints and preprocessing artifacts with Python pickle
semantics. This applies to:

- Kumo-Anomaly: checkpoints passed as `model_path` (`.pth`), loaded with `torch.load(..., weights_only=False)`.
- Kumo-Forecast: standardizers passed as `standardizer_pkl` (`.pkl`, loaded with `joblib.load`) and checkpoints passed as `ckpt` or `impute_ckpt` (`.pt`).

**Loading a crafted file can execute arbitrary code in your environment, before any validation runs.**
Only load checkpoints and standardizers that you created yourself or obtained from a source you trust,
such as the official [`nvidia/Kumo-Anomaly`](https://huggingface.co/nvidia/Kumo-Anomaly) and
[`nvidia/Kumo-Forecast`](https://huggingface.co/nvidia/Kumo-Forecast) Hugging Face repositories, which
the SDK downloads by default. Never load fine-tuned models or standardizers shared by unknown parties
or downloaded from untrusted locations. If you must inspect an untrusted file, do so in an isolated,
disposable environment.

Loading an untrusted file you supplied through these parameters is a documented usage limitation, not a
vulnerability in the SDK. To report a vulnerability, follow the process above.
