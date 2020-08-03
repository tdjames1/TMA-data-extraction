source("extract_data.R")

files <- list.files("../../Sample\ PDFs/", "*.pdf$", full.names = TRUE)

combined_data <- purrr::map_dfr(files, extract_data, out_dir = "working")

write.csv(combined_data, file = "working/alert_data_full.csv")

#####################################

# Example: "WARNING of heavy rain is issued for some areas of Tanga region
# together with Unguja and Pemba Isles. Likelihood: HIGH Impact: MEDIUM ADVISORY
# of heavy rain is issued for some areas of Dar es salaam, northern part of
# Lindi and Pwani (including Mafia isles) regions. Likelihood: MEDIUM Impact:
# MEDIUM Impacts expected: Some of the settlements to be surrounded by
# water/Flooding affecting parts of communities Please TAKE ACTION"

## Date
## Alert type: WARNING/ADVISORY/NO WARNING
## Alert text: "WARNING of heavy rain is issued for some areas of Tanga region
# together with Unguja and Pemba Isles."
## Weather type? e.g. heavy rain, strong winds
## Likelihood: XXX
## Impact: XXX
## Impacts expected: "Some of the settlements to be surrounded by water/Flooding
## affecting parts of communities" (N.B. not always associated with text
## immediately preceding - colour coding on )
## Advice: Please TAKE ACTION/Please BE PREPARED
