import mysql.connector
from mysql.connector import Error
import hashlib
import os

class Database:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="root",
                database="lungwatch"
            )
            print("Database connected successfully")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def __del__(self):
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print("Database connection closed")

    def create_tables(self):
        if not self.conn or not self.conn.cursor():
            raise Exception("Database connection not established")
            
        try:
            # Create users table with new fields
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    license_number VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(255),
                    specialization VARCHAR(100),
                    hospital VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
            print("Users table created/verified successfully")
        except Error as e:
            print(f"Error creating tables: {e}")
            raise

    def hash_password(self, password):
        """Hash a password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, license_number, name, password, email=None, specialization=None, hospital=None):
        try:
            # Check if user already exists
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE license_number = %s", (license_number,))
            if cursor.fetchone():
                return False, "License number already registered"

            # Hash the password
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            # Insert new user
            query = """
                INSERT INTO users (license_number, name, password, email, specialization, hospital)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            values = (license_number, name, hashed_password, email, specialization, hospital)
            
            print("Executing query with values:", values)  # Debug print
            cursor.execute(query, values)
            self.conn.commit()

            # Get the user data to return
            cursor.execute("""
                SELECT id, license_number, name, email, specialization, hospital
                FROM users WHERE license_number = %s
            """, (license_number,))
            user = cursor.fetchone()
            
            print("Successfully registered user:", user)  # Debug print
            return True, user

        except Error as e:
            print(f"Error in register_user: {e}")
            self.conn.rollback()
            return False, str(e)

    def login_user(self, license_number, password):
        try:
            # Hash the password
            hashed_password = hashlib.sha256(password.encode()).hexdigest()

            # Get user with all fields except password
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, license_number, name, email, specialization, hospital
                FROM users 
                WHERE license_number = %s AND password = %s
            """, (license_number, hashed_password))
            user = cursor.fetchone()
            
            if user:
                print("Login successful for user:", user)
                return True, user
            return False, "Invalid credentials"

        except Error as e:
            print(f"Error in login_user: {e}")
            return False, str(e)

    def check_license_exists(self, license_number):
        """Check if a license number already exists"""
        if not self.conn or not self.conn.cursor():
            raise Exception("Database connection not established")
            
        try:
            cursor = self.conn.cursor(dictionary=True)
            query = "SELECT COUNT(*) as count FROM users WHERE license_number = %s"
            cursor.execute(query, (license_number,))
            result = cursor.fetchone()
            return result['count'] > 0
        except Error as e:
            print(f"Error checking license number: {e}")
            return False

