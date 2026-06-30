terraform {
  required_version = ">= 1.0"
  
  required_providers {
    elasticstack = {
      source  = "elastic/elasticstack"
      version = "~>0.14"
    }

    ec = {
      source  = "elastic/ec"
      version = "~> 0.12"
    }
  }
}

provider "ec" {
  apikey = var.elastic_cloud_api_key
}

resource "ec_elasticsearch_project" "demo_project" {
  region_id     = var.region
  name          = "demo_project"
  optimized_for = "general_purpose"
}

provider "elasticstack" {
  elasticsearch {
    endpoints = ["${ec_elasticsearch_project.demo_project.endpoints.elasticsearch}"]
    username  = ec_elasticsearch_project.demo_project.credentials.username
    password  = ec_elasticsearch_project.demo_project.credentials.password
  }
  alias = "demo_project"
}