# CI/CD Pipeline Setup Guide for Death Note Life Expectancy App

## Overview
This guide will help you set up a complete CI/CD pipeline using GitHub Actions to automatically test, build, and deploy your application to AWS EC2.

## Pipeline Stages

### 1. **Test Stage**
- Runs on every push and pull request
- Lints Python and JavaScript code
- Builds frontend to ensure no compilation errors
- Ensures code quality before deployment

### 2. **Build Stage**
- Builds Docker images for backend and frontend
- Validates that images build successfully
- Only runs on `main` branch

### 3. **Deploy Stage**
- Deploys to production EC2 instance
- Runs health checks
- Restarts Cloudflare tunnel
- Cleans up old Docker images

---

## GitHub Setup Instructions

### Step 1: Add Repository Secrets

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add the following secrets:

#### 1. **EC2_SSH_KEY**
```bash
# On your local machine or EC2, get the private key:
cat ~/.ssh/id_rsa
# OR if you have a specific key:
cat ~/.ssh/your-key-name.pem
```
- Copy the **entire private key** (including `-----BEGIN` and `-----END` lines)
- Paste it as the secret value

#### 2. **EC2_HOST**
```
# Your EC2 public IP or hostname
# Example:
ec2-XX-XXX-XXX-XX.compute-1.amazonaws.com
# OR
172.31.10.194
```

#### 3. **EC2_USERNAME**
```
# Usually 'ubuntu' for Ubuntu instances
ubuntu
```

---

### Step 2: Configure SSH Access on EC2

On your EC2 instance, ensure the GitHub Actions runner can SSH in:

```bash
# 1. Make sure your SSH key is in authorized_keys
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh

# 2. Test SSH access from another terminal:
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_EC2_IP
```

---

### Step 3: Set Up Git on EC2

```bash
# Configure Git (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"

# Ensure the repository is in the correct location
cd ~/deathnote
git remote -v
# Should show your GitHub repository URL

# Set up SSH for GitHub (recommended)
ssh-keygen -t ed25519 -C "your-email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add this to GitHub: Settings → SSH and GPG keys → New SSH key
```

---

### Step 4: Create Environment in GitHub (Optional but Recommended)

1. Go to **Settings** → **Environments** → **New environment**
2. Name it: `production`
3. Add protection rules:
   - ✅ Required reviewers (optional)
   - ✅ Wait timer (optional - e.g., 5 minutes)
4. Add environment secrets (same as repository secrets if needed)

---

### Step 5: Test the Pipeline

#### Option A: Push to Main Branch
```bash
git add .
git commit -m "Set up CI/CD pipeline"
git push origin main
```

#### Option B: Manual Trigger
1. Go to **Actions** tab in GitHub
2. Select **CI/CD Pipeline** workflow
3. Click **Run workflow** → **Run workflow**

---

## Monitoring Deployments

### View Workflow Runs
1. Go to **Actions** tab in your GitHub repository
2. Click on the latest workflow run
3. Monitor each job (Test → Build → Deploy)

### Check Logs on EC2
```bash
# View Docker logs
docker-compose logs -f

# View specific service logs
docker-compose logs backend --tail=100
docker-compose logs frontend --tail=50

# Check Cloudflare tunnel
sudo systemctl status cloudflared
sudo journalctl -u cloudflared -f
```

---

## Workflow Features

### ✅ Automatic Testing
- Runs Python linting (flake8, black)
- Runs npm linting
- Builds frontend to catch errors early

### ✅ Docker Image Building
- Builds fresh images on every main branch push
- Validates that all containers build successfully

### ✅ Zero-Downtime Deployment
- Pulls latest code from GitHub
- Rebuilds Docker images
- Stops old containers
- Starts new containers
- Runs health checks
- Restarts Cloudflare tunnel

### ✅ Health Checks
- Verifies backend is responding
- Verifies frontend is accessible
- Shows detailed logs if deployment fails

### ✅ Cleanup
- Removes old Docker images
- Saves disk space automatically

---

## Troubleshooting

### Problem: "Permission denied (publickey)"
**Solution:**
```bash
# Ensure EC2_SSH_KEY secret contains the correct private key
# Test SSH manually:
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_EC2_IP
```

### Problem: "Cannot pull from repository"
**Solution:**
```bash
# On EC2, set up SSH key for GitHub
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
# Add to GitHub SSH keys
```

### Problem: Docker build fails
**Solution:**
```bash
# Check Docker on EC2
docker ps
docker-compose ps

# Rebuild manually to see errors
docker-compose build --no-cache
```

### Problem: Health check fails
**Solution:**
```bash
# Check if services are running
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs frontend

# Check ports
sudo lsof -i :8000
sudo lsof -i :80
```

---

## Advanced Configuration

### Add Slack/Discord Notifications
Add this step at the end of the deploy job:

```yaml
- name: Notify Slack
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'Deployment to production ${{ job.status }}'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Deploy to Staging First
Create a separate workflow for staging:

```yaml
# .github/workflows/deploy-staging.yml
on:
  push:
    branches:
      - develop
```

### Run Database Migrations
Add before starting containers:

```bash
# In the deploy job
docker-compose run --rm backend python manage.py migrate
```

---

## Security Best Practices

1. ✅ Never commit secrets to the repository
2. ✅ Use GitHub Secrets for all sensitive data
3. ✅ Rotate SSH keys regularly
4. ✅ Use separate environments for staging/production
5. ✅ Enable branch protection on `main` branch
6. ✅ Require pull request reviews before merging

---

## Next Steps

1. **Set up branch protection:**
   - Settings → Branches → Add rule
   - Require pull request reviews
   - Require status checks to pass

2. **Add more tests:**
   - Unit tests for backend
   - Integration tests
   - E2E tests with Playwright/Cypress

3. **Add monitoring:**
   - Set up error tracking (Sentry)
   - Add application performance monitoring
   - Set up uptime monitoring

4. **Optimize builds:**
   - Use Docker layer caching
   - Add multi-stage builds
   - Use GitHub Actions cache

---

## Quick Reference

### Manual Deployment Commands
```bash
# SSH to EC2
ssh -i ~/.ssh/key.pem ubuntu@YOUR_EC2_IP

# Pull latest code
cd ~/deathnote
git pull origin main

# Rebuild and restart
docker-compose build --no-cache
docker-compose down
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

### Useful GitHub Actions Commands
```bash
# View all workflows
gh workflow list

# Run a workflow manually
gh workflow run ci-cd.yml

# View recent runs
gh run list

# View logs for a run
gh run view --log
```

---

## Support

If you encounter issues:
1. Check the **Actions** tab for detailed error logs
2. SSH into EC2 and check Docker logs
3. Verify all secrets are correctly set in GitHub
4. Ensure EC2 security groups allow SSH (port 22)

**Your pipeline is ready! 🚀**

Push to `main` branch or manually trigger the workflow to see it in action.
