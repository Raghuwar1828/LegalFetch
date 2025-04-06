<?php
session_start();
require_once "config.php"; // Reuse your existing database connection

// Initialize variables with default messages
$terms = "Enter a URL to see Terms of Service.";
$privacy = "Enter a URL to see Privacy Policy.";
$summary = "Enter a URL to see the 100-word summary.";
$one_line = "Enter a URL to see the single-line summary.";

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["generate"])) {
    $url = trim($_POST["url"]);

    // Fetch data from database
    $sql = "SELECT terms_of_service, privacy_policy, summary_100_words, one_line_summary 
            FROM company_policies 
            WHERE company_url = :url";
    $stmt = $conn->prepare($sql);
    $stmt->bindParam(":url", $url);
    $stmt->execute();

    if ($stmt->rowCount() == 1) {
        $row = $stmt->fetch(PDO::FETCH_ASSOC);
        $terms = $row["terms_of_service"];
        $privacy = $row["privacy_policy"];
        $summary = $row["summary_100_words"];
        $one_line = $row["one_line_summary"];
    } else {
        $terms = "No Terms of Service found for this URL.";
        $privacy = "No Privacy Policy found for this URL.";
        $summary = "No summary available for this URL.";
        $one_line = "No one-line summary available for this URL.";
    }
}
?>


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
    <script src="https://kit.fontawesome.com/e3831a00ca.js" crossorigin="anonymous"></script>
    <link rel="stylesheet" href="styles.css">
    <title>Home | LegalFetch</title>
</head>
<body>
    <div class="lf-home-container col-100">
        <div class="lf-home-main col-100 common flex-col">
            <div class="homepage col-100">
                <div class="navigation col-100 common-bet">
                    <div class="nav-logo col-20 common align">
                        <a href="home.php">LegalFetch</a>
                    </div>
                    <div class="nav-links col-40 common-even align">
                        <a href="home.php" id="active">Home</a>
                        <a href="contact.php">Contact</a>
                        <a href="profile.php">Profile</a>
                        <a href="logout.php" id="logout">Logout</a>
                    </div>
                </div>
                <div class="lf-main col-100 common-even align flex-col"style="background-image: linear-gradient(to right, #4a90e2, #8f11a8);margin-top: 80px;height: 100%;">
                    <h1 data-aos="zoom-in" data-aos-duration="900" >Extract and Analyze<br>Terms of Service<br>Effortlessly</h1>
                    <p data-aos="zoom-in" data-aos-duration="900" >Transform how you understand terms of service and privacy policies<br>with our innovative tool.</p>
                    <a href="#product" data-aos="zoom-in" data-aos-duration="900" >Get Started</a>
                </div>
            </div>
            <div class="product col-100" id="product">
                <div class="search col-100 common">
                    <form action="home.php" method="POST" class="col-50 common flex-col align">
                        <input type="search" id="searchBox" name="url" placeholder="Enter website URL here" required>
                        <div class="suggestions" id="suggestionsList"></div>
                        <input type="submit" value="Generate" name="generate">
                    </form>
                </div>
                <div class="display-tos col-100 common flex-col align">
                    <div class="display-tos-main col-80">
                        <div class="display-box col-100">
                            <h2>Terms of Service</h2>
                            <li><?php echo htmlspecialchars($terms ?? 'Enter a URL to see Terms of Service.'); ?></li>
                        </div>
                        <div class="display-box col-100">
                            <h2>Privacy Policy</h2>
                            <li><?php echo htmlspecialchars($privacy ?? 'Enter a URL to see Privacy Policy.'); ?></li>
                            
                        </div><br>
                    </div>
                    <div class="display-tos-main col-80">
                        <h2>100 word summary</h2>
                        <li><?php echo htmlspecialchars($summary ?? 'Enter a URL to see the 100-word summary.'); ?></li>
                        
                    </div>
                    <div class="display-tos-main col-80">
                        <h2>Single line summary</h2>
                        <li><?php echo htmlspecialchars($one_line ?? 'Enter a URL to see the single-line summary.'); ?></li>
                    </div>
                </div>
            </div>
        </div>
    </div>


    <script src="script.js"></script>
    <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
    <script>
        AOS.init();
    </script>
</body>
</html>