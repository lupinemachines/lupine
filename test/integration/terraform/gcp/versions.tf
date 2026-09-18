terraform {
  required_version = ">= 1.9, < 2.0"

  # Supply an existing bucket and a unique per-run prefix during terraform init.
  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "8.1.0"
    }
  }
}

provider "google" {
  project                         = var.project_id
  region                          = local.region
  zone                            = var.zone
  add_terraform_attribution_label = false
}
