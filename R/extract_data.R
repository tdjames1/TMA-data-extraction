##' Extract data from Tanzanian Meteorological Authority
##' "Five days Severe weather impact-based forecasts" PDF
##'
##' This function locates text information relating to severe weather forecasts
##' in PDF files issued by the Tanzanian Meteorological Authority.  The function
##' returns details for each of 4 days following the date of issue as a tibble
##' containing issue date, forecast date (as day [in sequence], weekday and
##' date), alert text, and (x, y) position of alert column heading.
##'
##' If `out_dir` is specified then both the raw data and alert data are stored
##' to the specified directory in CSV format with a file name prefixed by the
##' alert issue date in ISO 8601 format.
##'
##' @title Extract TMA weather alert data from PDF
##' @param file_path Path to PDF file
##' @param out_dir [Optional] Output directory for storing data for individual files
##' @return Tibble containing alert details for 4 days following the date of issue
##'
##' @examples
##' \dontrun{
##' extract_data("test.pdf", "working")
##' }
##'
##' @author Tamora James, \email{T.D.James1@leeds.ac.uk}
extract_data <- function(file_path, out_dir = NULL) {

    if (!is.null(out_dir) && !dir.exists(out_dir)) {
        dir.create(out_dir)
    }

    ## Extract raw text, find issue date
    text <- pdftools::pdf_text(file_path)
    pos <- stringr::str_locate(text, "Issued on")
    if (is.na(pos[1,"start"])) {
        stop("Couldn't find date of issue")
    }

    ## Issued on Tuesday: 28-04-2020 at 15:30 (EAT)
    re <- "Issued on ([A-Za-z]*): ([0-9]{2}-[0-9]{2}-[0-9]{4}) at ([0-9]{2}:[0-9]{2})"
    matches <- stringr::str_match(text, re)[1,-1]  # Extract matches from page 1
    ## issue_day <- matches[1]
    ## issue_date <- matches[2]
    ## issue_time <- matches[3]

    issue_date <- as.Date(matches[2], format = "%d-%m-%Y")
    message(paste("Severe weather warnings issued on:", issue_date))

    if (!is.null(out_dir)) {
        file_prefix <- paste(out_dir, format(issue_date, format = "%Y-%m-%d"), sep = "/")
    }

    ## Extract positioned text, search for position of alert columns
    data <- pdftools::pdf_data(file_path)

    ## Combine page data and save as csv
    if (!is.null(out_dir)) {
        pdf_data <- purrr::imap_dfr(data, ~dplyr::mutate(.x, page = .y))
        write.csv(pdf_data, paste(file_prefix, "raw_data.csv", sep = "_"))
    }

    ## Get positions of columns
    pos <- dplyr::filter(data[[2]], text %in% weekdays(issue_date + 1:4))
    pos <- dplyr::mutate(pos, diff_x = c(diff(x), Inf))

    ## Get y position of targeted string
    locate_text_y <- function(data, text) {
        locate_text_vert <- function(df, words) {
            loc <- dplyr::filter(df, text %in% words)
            ypos <- rle(loc$y)
            ypos <- ifelse(length(ypos$values) > 0,
                           ypos$values[ypos$lengths == length(words)],
                           NA)
        }
        words <- unlist(stringr::str_split(text, " "))
        if (is.data.frame(data)) {
            locate_text_vert(data, words)
        } else {
            purrr::map(data, ~locate_text_vert(.x, words))
        }
    }

    ## Extract column text
    get_column_text <- function(x, y, diff_x, page_data, min_y = NULL, max_y = Inf) {
        col_text <- purrr::pmap(list(x, y, diff_x), function(pos_x, pos_y, diff_x) {
            if (is.null(min_y)) {
                min_y <- pos_y
            }
            col <- dplyr::filter(page_data, x >= pos_x -5 & x < (pos_x + diff_x), y > min_y & y < max_y)
            return(paste(col$text, collapse = " "))
        })
        return(unlist(col_text))
    }

    ## Get alert text across pages
    get_alert_text <- function(data, pos_data) {
        header_text <- "Five days Severe weather impact-based forecasts"
        footer_text <- "Revision No. 01"
        use_ymin_loc <- FALSE
        alert_text <- purrr::map(data[2:length(data)], function(page_data) {
            if (use_ymin_loc) {
                ymin  <- pos_data$y
            } else {
                ymin <- locate_text_y(page_data, header_text)
            }
            ymax <- locate_text_y(page_data, footer_text)
            get_column_text(pos_data$x, pos_data$y, pos_data$diff_x,
                            page_data, min_y = ymin, max_y = ymax)
            })
        alert_text <- purrr::reduce(alert_text, mapply, FUN = paste,
                                    SIMPLIFY = FALSE, USE.NAMES = FALSE)
        return(unlist(alert_text))
    }

    ## Build output data frame
    alert_data <- tibble::tibble(issue_date = issue_date,
                                 day = 1:4,
                                 weekday = weekdays(issue_date + day),
                                 date = issue_date + day,
                                 alert_text = get_alert_text(data, pos_data = pos))
    alert_data <- dplyr::left_join(alert_data,
                                   dplyr::select(pos, x, y, text),
                                   by = c("weekday" = "text"))

    if (!is.null(out_dir)) {
        write.csv(alert_data, paste(file_prefix, "alert_data.csv", sep = "_"))
    }

    return(alert_data)
}
