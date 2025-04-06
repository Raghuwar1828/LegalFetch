<?php
session_start();
require_once "config.php"; // Database connection

// Check if user is logged in
if (!isset($_SESSION["loggedin"]) || $_SESSION["loggedin"] !== true) {
    header("Location: signin.php"); // Redirect to sign-in if not logged in
    exit;
}
// Fetch user data from database
$user_id = $_SESSION["id"];
$sql = "SELECT full_name, email, created_at FROM users WHERE id = :id";
$stmt = $conn->prepare($sql);
$stmt->bindParam(":id", $user_id, PDO::PARAM_INT);
$stmt->execute();
$user = $stmt->fetch(PDO::FETCH_ASSOC);

if (!$user) {
    // If no user found (unlikely with session check), log out and redirect
    session_destroy();
    header("Location: signin.html");
    exit;
}

// Assign user data to variables
$full_name = $user["full_name"];
$email = $user["email"];
$created_at = $user["created_at"];
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
    <script src="https://kit.fontawesome.com/e3831a00ca.js" crossorigin="anonymous"></script>
    <link rel="stylesheet" href="styles.css">
    <title>Profile | LegalFetch</title>
</head>
<body>
    <div class="ls-home-container col-100">
        <div class="ls-home-main col-100">
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
                <div class="profile-main col-100 common align">
                    <div class="profile-card common align flex-col">
                        <img src="resources/profile.png" alt="profile">
                        <span class="common"><b>Name:</b><p> <?php echo htmlspecialchars($full_name); ?></p></span>
                        <span class="common"><b>Email:</b><p> <?php echo htmlspecialchars($email); ?></p></span>
                        <span class="common"><b>Account created at:</b><p> <?php echo htmlspecialchars($created_at); ?></p></span>
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