output "inventory" {
  description = "Provisioned machine addresses and expected hardware. The runner must discover actual GPU UUIDs and capabilities."
  value = {
    version = 1
    run_id  = var.run_id
    hosts = {
      for role, host in google_compute_instance.host : role => {
        instance_id        = host.instance_id
        name               = host.name
        zone               = host.zone
        ssh_host           = host.network_interface[0].access_config[0].nat_ip
        ssh_user           = var.ssh_user
        rpc_host           = host.network_interface[0].network_ip
        machine_type       = local.hosts[role].machine_type
        os                 = local.hosts[role].os
        version            = local.hosts[role].version
        arch               = local.hosts[role].arch
        expected_gpu_count = local.hosts[role].gpu_count
        expected_gpu_model = local.hosts[role].gpu_count == 0 ? "" : "L4"
        image              = host.boot_disk[0].initialize_params[0].image
      }
    }
  }
}
