db = db.getSiblingDB("admin"); // Switch to admin database

db.createUser({
  user: "shreyas",
  pwd: "devpassword",
  roles: [
    { role: "root", db: "admin" }, // Root access in admin DB
  ],
});

print("Admin user 'shreyas' created successfully!");

db = db.getSiblingDB("binge_buddy_db"); // Switch to binge_buddy_db

db.createUser({
  user: "shreyas",
  pwd: "devpassword",
  roles: [
    { role: "readWrite", db: "binge_buddy_db" }, // Read/Write in binge_buddy_db
  ],
});

print("User 'shreyas' created successfully in binge_buddy_db!");
