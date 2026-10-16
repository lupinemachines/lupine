variable "project_id" {
  description = "Existing GCP project with Compute Engine enabled and sufficient GPU quota."
  type        = string
}

variable "run_id" {
  description = "Unique run name, also used as the resource prefix and network tag."
  type        = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,38}[a-z0-9]$", var.run_id))
    error_message = "Use 2-40 lowercase letters, digits or hyphens, beginning with a letter and ending with a letter or digit."
  }
}

variable "zone" {
  description = "Zone with capacity for the two g2-standard-24 L4 VMs."
  type        = string
  default     = "us-central1-a"
}

variable "network" {
  description = "Existing auto-mode network used by the GPU CI service account."
  type        = string
  default     = "default"
}

variable "host_platforms" {
  description = "OS, architecture and pinned image per host, resolved from runs.yaml. GPU images must include L4 drivers."
  type = map(object({
    os           = string
    version      = string
    arch         = string
    machine_type = string
    image        = string
  }))
  validation {
    condition = toset(keys(var.host_platforms)) == toset(["client", "a", "b"]) && alltrue([
      for role, platform in var.host_platforms :
      platform.os == "ubuntu" && contains(["22.04", "24.04"], platform.version) &&
      contains(["x86_64", "arm64"], platform.arch) &&
      can(regex("^projects/[^/]+/global/images/[^/]+$", platform.image)) &&
      (role == "client" ?
        platform.machine_type == lookup({ arm64 = "t2a-standard-4", x86_64 = "e2-standard-4" }, platform.arch, "") :
      platform.arch == "x86_64" && platform.machine_type == "g2-standard-24")
    ])
    error_message = "Assign supported Ubuntu platforms and concrete images to client, a and b; GPU hosts must remain x86_64 G2 L4 VMs."
  }
}

variable "ssh_user" {
  type    = string
  default = "lupine"
}

variable "ssh_public_key" {
  description = "Public key for the runner; the private key stays outside Terraform."
  type        = string
}

variable "ssh_source_ranges" {
  description = "IPv4 CIDRs allowed to SSH to the fleet, normally the CI runner's egress IP /32."
  type        = list(string)
  validation {
    condition     = length(var.ssh_source_ranges) > 0 && alltrue([for cidr in var.ssh_source_ranges : can(cidrnetmask(cidr))])
    error_message = "Provide at least one valid IPv4 CIDR for runner SSH access."
  }
}

variable "max_run_seconds" {
  description = "VM expiry fallback. Terraform destroy must still remove network resources."
  type        = number
  default     = 15600
  validation {
    condition     = var.max_run_seconds >= 60 && floor(var.max_run_seconds) == var.max_run_seconds
    error_message = "The maximum runtime must be an integer of at least 60 seconds."
  }
}

variable "reservations" {
  description = "Optional map from GPU host role (a or b) to an existing matching reservation in the selected zone."
  type        = map(string)
  default     = {}
  validation {
    condition     = alltrue([for role, name in var.reservations : contains(["a", "b"], role) && length(name) > 0])
    error_message = "Reservations must name GPU host a or b and a nonempty reservation name."
  }
}
