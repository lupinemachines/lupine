locals {
  region  = join("-", slice(split("-", var.zone), 0, length(split("-", var.zone)) - 1))
  network = "projects/${var.project_id}/global/networks/${var.network}"
  hosts = {
    client = { machine_type = "e2-standard-4", gpu_count = 0 }
    a      = { machine_type = "g2-standard-24", gpu_count = 2 }
    b      = { machine_type = "g2-standard-24", gpu_count = 2 }
  }
}

resource "google_compute_firewall" "ssh" {
  name          = "${var.run_id}-ssh"
  network       = local.network
  priority      = 800
  source_ranges = var.ssh_source_ranges
  target_tags   = [var.run_id]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_firewall" "rpc" {
  name        = "${var.run_id}-rpc"
  network     = local.network
  priority    = 800
  source_tags = [var.run_id]
  target_tags = [var.run_id]
  allow {
    protocol = "tcp"
    ports    = ["14833-14932"]
  }
}

resource "google_compute_firewall" "deny" {
  name          = "${var.run_id}-deny"
  network       = local.network
  priority      = 900
  source_ranges = ["0.0.0.0/0"]
  target_tags   = [var.run_id]
  deny {
    protocol = "all"
  }
}

resource "google_compute_instance" "host" {
  for_each     = local.hosts
  name         = "${var.run_id}-${each.key}"
  machine_type = each.value.machine_type
  zone         = var.zone
  tags         = [var.run_id]
  depends_on   = [google_compute_firewall.ssh, google_compute_firewall.rpc, google_compute_firewall.deny]

  boot_disk {
    auto_delete = true
    initialize_params {
      image = each.value.gpu_count == 0 ? var.cpu_image : var.gpu_image
      size  = 100
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = local.network
    # SSH uses this address; Lupine uses network_ip within the test VPC.
    access_config {}
  }

  metadata = {
    block-project-ssh-keys = "TRUE"
    enable-oslogin         = "FALSE"
    ssh-keys               = "${var.ssh_user}:${trimspace(var.ssh_public_key)}"
  }
  metadata_startup_script = file("${path.module}/bootstrap.sh")

  scheduling {
    provisioning_model          = "STANDARD"
    on_host_maintenance         = "TERMINATE"
    automatic_restart           = false
    instance_termination_action = "DELETE"
    max_run_duration {
      seconds = var.max_run_seconds
    }
  }

  reservation_affinity {
    type = contains(keys(var.reservations), each.key) ? "SPECIFIC_RESERVATION" : "NO_RESERVATION"
    dynamic "specific_reservation" {
      for_each = contains(keys(var.reservations), each.key) ? [var.reservations[each.key]] : []
      content {
        key    = "compute.googleapis.com/reservation-name"
        values = [specific_reservation.value]
      }
    }
  }

  shielded_instance_config {
    enable_secure_boot          = false
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }
}
