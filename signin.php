<?php
session_start();
require_once "config.php";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = trim($_POST["mailid"]);
    $password = trim($_POST["password"]);

    // Check if user exists
    $sql = "SELECT id, full_name, email, password FROM users WHERE email = :email";
    $stmt = $conn->prepare($sql);
    $stmt->bindParam(":email", $email);
    $stmt->execute();

    if ($stmt->rowCount() == 1) {
        $user = $stmt->fetch(PDO::FETCH_ASSOC);
        if (password_verify($password, $user["password"])) {
            // Password is correct, start session
            $_SESSION["loggedin"] = true;
            $_SESSION["id"] = $user["id"];
            $_SESSION["full_name"] = $user["full_name"];
            $_SESSION["email"] = $user["email"];
            header("Location: profile.php"); // Redirect to profile page (create this later)
            exit;
        } else {
            echo "Invalid password.";
        }
    } else {
        echo "No account found with that email.";
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
    <title>Sign In | LegalFetch</title>
</head>
<body>
    <div class="lf-container col-100 common align">
        <a href="index.html" id="tempLogo">LegalFetch</a>
        <div class="lf-signin col-100">
            <div class="signin-form common align" data-aos="fade-up" data-aos-duration="1000">
                <form action="signin.php" method="POST" class="common flex-col">
                    <h3>Sign In</h3>
                    <label for="mailid">Email</label>
                    <input type="email" id="mailid" name="mailid" placeholder="Your Email here" required>
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" placeholder="Your Password here" required>
                    <input type="submit" name="signin" id="signin" value="Sign In"><br>
                    <span>No account? <a href="signup.php">Sign Up</a></span>
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