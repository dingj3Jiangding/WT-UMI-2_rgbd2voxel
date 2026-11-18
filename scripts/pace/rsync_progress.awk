# Render native rsync progress on a terminal without inventing completion events.
# A single-character separator avoids macOS awk's regex lookahead buffering.
# Newline records are handled inside each carriage-return-delimited update.
BEGIN {
    RS = "\r"
    width = 24
    active = 0
    if (label == "") label = "Current file"
}

function finish_line() {
    if (active) {
        printf "\n"
        active = 0
    }
}

function render_line(line) {
    $0 = line
    if ($1 ~ /^[0-9][0-9,.]*[kKMGTPE]?$/ && $2 ~ /^[0-9]+%$/ &&
        $3 ~ /\/s$/ && $4 ~ /^[0-9]+:[0-9]+:[0-9]+$/) {
        percent = $2 + 0
        if (percent < 0) percent = 0
        if (percent > 100) percent = 100
        filled = int(percent * width / 100)
        bar = ""
        for (i = 0; i < width; i++) bar = bar (i < filled ? "#" : "-")
        printf "\r%s [%s] %3d%%  %s  %s\033[K", label, bar, percent, $3, $4
        active = 1
        fflush()
    } else if (length($0)) {
        finish_line()
        print $0
        fflush()
    }
}

{
    count = split($0, lines, "\n")
    for (line_index = 1; line_index <= count; line_index++) render_line(lines[line_index])
}

END {
    finish_line()
}
