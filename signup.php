<?php
session_start();
require_once "config.php";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $full_name = trim($_POST["fname"]);
    $email = trim($_POST["mailid"]);
    $password = password_hash(trim($_POST["password"]), PASSWORD_DEFAULT); // Hash the password

    // Check if email already exists
    $sql = "SELECT id FROM users WHERE email = :email";
    $stmt = $conn->prepare($sql);
    $stmt->bindParam(":email", $email);
    $stmt->execute();

    if ($stmt->rowCount() > 0) {
        echo "Email already exists. Please use a different email.";
    } else {
        // Insert new user
        $sql = "INSERT INTO users (full_name, email, password) VALUES (:full_name, :email, :password)";
        $stmt = $conn->prepare($sql);
        $stmt->bindParam(":full_name", $full_name);
        $stmt->bindParam(":email", $email);
        $stmt->bindParam(":password", $password);

        if ($stmt->execute()) {
            header("Location: signin.php"); // Redirect to sign-in page
            exit;
        } else {
            echo "Something went wrong. Please try again.";
        }
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
    <title>Sign Up | LegalFetch</title>
</head>
<body>
    <div class="lf-container col-100 common align">
        <a href="index.html" id="tempLogo">LegalFetch</a>
        <div class="lf-signin col-100">
            <div class="signin-form common align" data-aos="fade-up" data-aos-duration="1000">
                <form action="signup.php" method="POST" class="common flex-col">
                    <h3>Sign Up</h3>
                    <label for="fname">Full Name</label>
                    <input type="text" name="fname" id="fname" placeholder="Your Full name here" required>
                    <label for="mailid">Email</label>
                    <input type="email" id="mailid" name="mailid" placeholder="Your Email here" required>
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" placeholder="Your Password here" required>
                    <input type="submit" name="signin" id="signin" value="Sign Up"><br>
                    <span>Already have an account? <a href="signin.php">Sign In</a></span>
                </form>
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