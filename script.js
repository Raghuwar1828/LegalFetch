const searchBox = document.getElementById("searchBox");
        const suggestionsList = document.getElementById("suggestionsList");

        const suggestedURLs = [
            "google.com",
            "apple.com",
            "microsoft.com",
            "amazon.com",
            "meta.com"
        ];

        searchBox.addEventListener("input", function() {
            const input = searchBox.value.toLowerCase();
            suggestionsList.innerHTML = "";

            if (input) {
                const filteredURLs = suggestedURLs.filter(url => url.includes(input));

                if (filteredURLs.length > 0) {
                    suggestionsList.style.display = "block";
                    filteredURLs.forEach(url => {
                        const div = document.createElement("div");
                        div.textContent = url;
                        div.addEventListener("click", () => {
                            searchBox.value = url;
                            suggestionsList.style.display = "none";
                        });
                        suggestionsList.appendChild(div);
                    });
                } else {
                    suggestionsList.style.display = "none";
                }
            } else {
                suggestionsList.style.display = "none";
            }
        });

        document.addEventListener("click", (e) => {
            if (!searchBox.contains(e.target) && !suggestionsList.contains(e.target)) {
                suggestionsList.style.display = "none";
            }
        });