output "elastic_cloud_id" {
  value = ec_elasticsearch_project.demo_project.cloud_id
}

output "elastic_cloud_api_key" {
  value = var.elastic_cloud_api_key
  sensitive = true
}

output "hf_token" {
  value = var.hf_token
  sensitive = true
}
