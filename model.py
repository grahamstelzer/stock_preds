function exportTableToCSV(tableClassName, filename) {
    let csv = [];
    let table = document.querySelector(`table.${tableClassName.replace(/\s+/g, '.')}`);
    
    // Check if the table is found
    if (!table) {
        console.error("Table not found for the specified class.");
        return;
    }

    // Capture headers
    let headers = table.querySelectorAll("th");
    let headerRow = [];
    headers.forEach(header => {
        let headerText = header.innerText.trim().replace(/,/g, '');
        headerRow.push(headerText);
    });
    csv.push(headerRow.join(","));

    // Capture data rows
    let rows = table.querySelectorAll("tbody tr");
    rows.forEach(row => {
        let rowData = [];
        let cols = row.querySelectorAll("td");

        cols.forEach((col, index) => {
            let cellText = col.innerText.trim().replace(/,/g, '');

            // Transform Date column from "Oct 23 2024" to "YYYY-MM-DD"
            if (headers[index].innerText.includes("Date")) {
                let dateParts = cellText.split(' ');
                let month = new Date(`${dateParts[0]} 1, 2020`).getMonth() + 1;
                let day = dateParts[1].replace(',', '');
                let year = dateParts[2];
                cellText = `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
            }

            rowData.push(cellText);
        });

        csv.push(rowData.join(","));
    });

    // Create a CSV file blob
    let csvFile = new Blob([csv.join("\n")], { type: "text/csv" });

    // Create a download link
    let downloadLink = document.createElement("a");
    downloadLink.download = filename;
    downloadLink.href = window.URL.createObjectURL(csvFile);

    // Append the link to the body and simulate a click to download
    document.body.appendChild(downloadLink);
    downloadLink.click();

    // Remove the download link after clicking
    document.body.removeChild(downloadLink);
}

// Call the function with the table class name
exportTableToCSV(tableClassName = 'table yf-j5d1ld', filename = 'output.csv');