from typing import List, Tuple
from datetime import datetime
import sqlite3


class PullRequestDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS pull_requests (
                                  id INTEGER PRIMARY KEY,
                                  pr_number INTEGER NOT NULL,
                                  summary TEXT NOT NULL,
                                  date DATE NOT NULL)"""
            )
            conn.commit()

    def add_pull_request(self, pr_number: int, summary: str, date: datetime):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO pull_requests (pr_number, summary, date) VALUES (?, ?, ?)",
                (pr_number, summary, date.date()),
            )
            conn.commit()

    def get_pull_request(self, pr_number: int) -> Tuple[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT summary FROM pull_requests WHERE pr_number = ?", (pr_number,))
            result = cursor.fetchone()
            if result:
                return result
            return None

    def get_all_pull_requests(self) -> List[Tuple[int, str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pr_number, summary, date FROM pull_requests")
            results = cursor.fetchall()
            return results

    def get_pull_requests_by_date(self, date: datetime) -> List[Tuple[int, str, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pr_number, summary, date FROM pull_requests WHERE date = ?", (date.date(),))
            results = cursor.fetchall()
            return results

    def delete_pull_request(self, pr_number: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pull_requests WHERE pr_number = ?", (pr_number,))
            conn.commit()


# Example usage
if __name__ == "__main__":
    db = PullRequestDatabase("pull_requests.db")
    db.add_pull_request(123, "This is a summary for pull request #123.")
    db.add_pull_request(124, "This is a summary for pull request #124.")

    print("Pull request #123:")
    print(db.get_pull_request(123))

    print("\nAll pull requests:")
    print(db.get_all_pull_requests())

    db.delete_pull_request(123)

    print("\nAll pull requests after deleting #123:")
    print(db.get_all_pull_requests())
