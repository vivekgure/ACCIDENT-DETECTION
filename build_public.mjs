import fs from "node:fs";

fs.rmSync("public_dist", { recursive: true, force: true });
fs.mkdirSync("public_dist", { recursive: true });
fs.copyFileSync("templates/index.html", "public_dist/index.html");

for (const dir of ["static", "sample_data", "accident_events"]) {
  fs.cpSync(dir, `public_dist/${dir}`, { recursive: true });
}
console.log("Built public_dist successfully.");
