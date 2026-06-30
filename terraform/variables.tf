variable "elastic_cloud_api_key" {
  description = "Elastic Cloud API key with Cloud and Stack access"
  type        = string
  sensitive   = true
}

variable "hf_token" {
  description = "Hugging Face API token"
  type        = string
  sensitive   = true
}

variable "region" {
  description = "GCP region"
  type        = string  
  default     = "gcp-us-central1"
}