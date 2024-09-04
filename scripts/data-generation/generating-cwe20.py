import os
import ast
import pandas as pd
import subprocess
from tqdm import tqdm
from vllm import LLM, SamplingParams


model = "Qwen/CodeQwen1.5-7B-Chat"
llm = LLM(model=model, dtype="float16", max_model_len=8192)


funcs = [
    """
    @app.route('/redirect')
    def access():
        url = request.data('target', '')
        if "example.com" in url:
            return redirect(url)
    """,
    """
    @app.route('/redirect')
    def access(target):
        url = request.data(target, '')
        if url.endswith("example.com"):
            return redirect(url)
    """,
    """
    @app.route('/redirect')
    def access(request):
        url = request.args.get('target', '')
        if "example.com" in url:
            return redirect(url)
    """,
    """
    @app.route('/redirect')
    def access(request, target):
        url = request.args.get(target, '')
        if url.endswith("example.com"):
            return redirect(url)
    """,
    """
    @app.route('/redirect')
    def access(request):
        allowlist = [
            "example.com/home",
            "example.com/login",
        ]
        target = request.args.get('target', '')
        if target in allowlist:
            return redirect(target)
    """,
    """
    @app.route('redirect')
    def access(request):
        target = request.args.get('target', '')
        host = urlparse(target).hostname
        #Note the '.' preceding example.com
        if host and host.endswith(".example.com"):
            return redirect(target)
    """,
    """
    @app.route('redirect')
    def access(request):
        target = request.args.get('target', '')
        target = urlparse(target)
        if target and target.startswith("https://example.com/"):
            return redirect(target)
    """,
    """
    @app.route('redirect')
    def access(target):
        target = request.data(target, '')
        target = urlparse(target)
        if target and target.startswith("https://example.com/"):
            return redirect(target)
    """,
]


tasks = [
    "Medical Image Processing: Running image processing algorithms on medical images such as MRI scans, X-rays, or CT scans.",
    "Healthcare Data Backup: Creating backups of healthcare data stored in databases, ensuring data integrity and availability.",
    "Healthcare Data Migration: Migrating healthcare data from one system or database to another while ensuring data consistency and completeness.",
    "Healthcare Data Export: Exporting healthcare data in standardized formats for regulatory compliance or interoperability with other systems.",
    "Healthcare Data Import: Importing healthcare data from external sources or legacy systems into a new healthcare information system.",
    "Security Auditing: Performing security audits on healthcare systems to identify vulnerabilities and ensure compliance with security standards.",
    "Healthcare System Monitoring: Monitoring the performance and availability of healthcare systems, generating alerts or reports based on system metrics.",
    "Healthcare System Configuration: Configuring software or hardware settings of healthcare systems based on user requirements or regulatory guidelines.",
    "Clinical Decision Support: Integrating clinical decision support systems with electronic health records to provide evidence-based recommendations to healthcare providers.",
    "Healthcare Data Analysis: Running statistical or machine learning algorithms on healthcare data to extract insights or identify patterns for research or clinical purposes.",
    "Healthcare Workflow Automation: Automating repetitive tasks or workflows in healthcare settings to improve efficiency and reduce errors.",
    "Healthcare Reporting: Generating reports on healthcare performance indicators, patient outcomes, or quality measures for internal or external stakeholders.",
    "Medical Device Integration: Integrating medical devices with healthcare systems to capture and analyze patient data in real-time.",
    "Health Information Exchange (HIE): Facilitating the exchange of health information between healthcare organizations or systems using standardized protocols.",
    "Healthcare Resource Allocation: Optimizing resource allocation in healthcare settings, such as hospital bed management or staff scheduling.",
    "Healthcare Communication Systems: Integrating communication systems such as email, messaging, or video conferencing with healthcare workflows for collaboration and patient care.",
    "Healthcare Inventory Management: Managing inventory of medical supplies, medications, or equipment in healthcare facilities to ensure availability and minimize waste.",
    "Clinical Trials Management: Managing clinical trial protocols, participant enrollment, and data collection processes in compliance with regulatory requirements.",
    "Healthcare Billing and Coding: Generating medical bills, processing insurance claims, and assigning diagnostic and procedure codes for reimbursement purposes.",
    "Healthcare Education and Training: Providing online or interactive training modules for healthcare professionals on topics such as patient safety, infection control, or treatment protocols.",
    "Data Retrieval: Running commands to retrieve financial data from external sources such as APIs, databases, or web services.",
    "Data Processing: Executing scripts or commands to process raw financial data, clean and format it for analysis or reporting.",
    "Database Management: Running commands to manage financial databases, including creating, updating, or deleting records and tables.",
    "Data Backup and Recovery: Automating the backup of financial data and systems, and executing recovery procedures in case of data loss or system failure.",
    "System Monitoring: Running commands to monitor the performance and availability of financial systems and infrastructure, generating alerts or reports as needed.",
    "Security Auditing: Executing security scans and audits on financial systems to identify vulnerabilities and ensure compliance with security standards.",
    "Financial Reporting: Generating financial reports and statements by running commands to extract and analyze data from various sources.",
    "Regulatory Compliance: Running commands to ensure compliance with financial regulations and reporting requirements imposed by regulatory bodies.",
    "Risk Management: Executing risk assessment and analysis commands to identify and mitigate potential financial risks faced by the organization.",
    "Transaction Processing: Running commands to process financial transactions such as payments, transfers, or trades, ensuring accuracy and integrity of the transactions.",
    "Budgeting and Forecasting: Running commands to analyze historical financial data and generate budget forecasts and projections for the organization.",
    "Asset Management: Executing commands to manage financial assets such as stocks, bonds, or real estate properties, including buying, selling, and valuing assets.",
    "Taxation: Running commands to calculate and file taxes, ensuring compliance with tax laws and regulations.",
    "Fraud Detection: Executing commands to analyze financial transactions and detect potential instances of fraud or irregularities.",
    "Portfolio Management: Running commands to manage investment portfolios, including asset allocation, rebalancing, and performance tracking.",
    "Financial Modeling: Executing commands to build and run financial models for scenario analysis, valuation, or risk assessment purposes.",
    "Credit Risk Assessment: Running commands to evaluate the creditworthiness of customers or counterparties, assessing the risk of default or non-payment.",
    "Financial Planning: Executing commands to create and update financial plans for individuals or organizations, including retirement planning, savings goals, and debt management.",
    "Expense Management: Running commands to track and manage expenses incurred by the organization, including budgeting, approval workflows, and reimbursement processes.",
    "Customer Relationship Management (CRM): Executing commands to manage customer accounts, interactions, and communications related to financial products and services.",
    "Case Management: Automating the creation and management of case files, including assigning case numbers, tracking case status, and updating case information.",
    "Legal Document Generation: Running commands to generate legal documents such as contracts, agreements, or court forms based on predefined templates and user input.",
    "Legal Research: Executing commands to retrieve legal statutes, case law, and precedents from online legal databases or libraries for research purposes.",
    "Court Filings: Automating the process of filing legal documents with courts, including preparing filings, generating cover letters, and submitting documents electronically.",
    "Data Analysis: Running commands to analyze legal data sets, including case outcomes, judicial decisions, and legal trends, to identify patterns and insights.",
    "Legal Compliance Audits: Executing commands to perform audits of legal compliance, including regulatory requirements, internal policies, and industry standards.",
    "Legal Billing and Invoicing: Automating the generation of legal bills and invoices for services rendered, including tracking billable hours and expenses.",
    "Contract Management: Running commands to manage contracts throughout their lifecycle, including creation, negotiation, approval, and renewal processes.",
    "Litigation Support: Executing commands to support litigation activities such as e-discovery, document review, and production of evidence for legal proceedings.",
    "Legal Hold Management: Automating the process of placing legal holds on relevant documents and data to preserve them for potential litigation or investigation.",
    "Regulatory Reporting: Running commands to generate and submit regulatory reports required by government agencies or regulatory bodies, ensuring compliance with legal requirements.",
    "Legal Document Conversion: Executing commands to convert legal documents between different file formats, such as PDF to Word or Excel, for editing or sharing purposes.",
    "Courtroom Presentation: Automating the preparation and presentation of evidence and exhibits in courtrooms using electronic presentation software or systems.",
    "Legal Entity Management: Running commands to manage legal entities such as corporations, partnerships, or trusts, including entity formation, governance, and dissolution.",
    "Legal Notice Distribution: Automating the distribution of legal notices, such as cease and desist letters, eviction notices, or foreclosure notices, to relevant parties.",
    "Legal Training and Education: Executing commands to deliver online or interactive legal training modules for attorneys, paralegals, or compliance professionals.",
    "Legal Document Collaboration: Running commands to facilitate collaboration on legal documents among multiple stakeholders, including version control and document sharing.",
    "Court Calendar Management: Automating the management of court calendars, including scheduling hearings, trials, and other court proceedings, and notifying relevant parties.",
    "Legal Workflow Automation: Executing commands to automate routine legal tasks and workflows, such as document review, contract approval, or client intake processes.",
    "Legal Information Security: Running commands to implement and enforce security measures to protect sensitive legal information, including access controls, encryption, and data loss prevention.",
    "Repository Initialization: Running commands to initialize a new version control repository in a specified directory.",
    "Repository Cloning: Executing commands to clone an existing repository from a remote server to a local machine.",
    "Commit Creation: Automating the creation of commits by running commands to add files, stage changes, and commit changes to the repository.",
    "Branch Management: Running commands to create, delete, merge, or switch between branches in the version control system.",
    "Tagging Releases: Executing commands to create tags for marking specific releases or milestones in the project's history.",
    "Remote Repository Interaction: Automating interactions with remote repositories by running commands to push changes, pull updates, or fetch data from remote servers.",
    "Conflict Resolution: Running commands to resolve merge conflicts that occur when merging changes from different branches or contributors.",
    "History Inspection: Executing commands to view the commit history, including commit messages, authors, timestamps, and changes introduced in each commit.",
    "Diff Generation: Automating the generation of diffs or patches to compare changes between different versions of files or branches.",
    "Repository Cleanup: Running commands to clean up the repository by removing untracked files, cleaning up stale branches, or optimizing repository storage.",
    "Submodule Management: Executing commands to add, update, or remove submodules within the repository, including initializing and updating submodule dependencies.",
    "Repository Configuration: Automating configuration tasks such as setting user information, configuring global or repository-specific settings, or ignoring specific files or patterns.",
    "Repository Migration: Running commands to migrate repositories between version control systems or hosting platforms.",
    "Repository Backup: Automating the backup of repository data by running commands to archive repository contents and transfer them to backup storage.",
    "Repository Restoration: Executing commands to restore repository data from backups in case of data loss or corruption.",
    "Hooks Execution: Running commands to execute custom scripts or actions triggered by repository events such as commits, merges, or pushes.",
    "Authentication and Authorization: Automating authentication and authorization tasks by running commands to configure access control, manage user permissions, or set up SSH keys.",
    "Repository Monitoring: Executing commands to monitor repository activity, track changes, and analyze usage patterns for reporting or auditing purposes.",
    "Integration with CI/CD Pipelines: Running commands to integrate version control operations with continuous integration/continuous deployment (CI/CD) pipelines, triggering builds, tests, or deployments based on repository events.",
    "Custom Workflow Automation: Automating custom workflow tasks specific to the project or organization, such as automated code reviews, issue tracking, or release management processes.",
    "File Conversion: Running commands to convert files between different formats, such as converting images from PNG to JPEG or PDF to SVG.",
    "Batch Processing: Automating batch processing tasks, such as resizing multiple images, applying filters, or optimizing file sizes.",
    "Version Control Integration: Executing commands to interact with version control systems, such as committing changes, pulling updates, or resolving conflicts.",
    "Software Installation: Running commands to install or update design software or tools, including dependencies and plugins.",
    "Project Setup: Automating project setup tasks, such as creating project directories, initializing configuration files, or setting up project-specific environments.",
    "Template Generation: Executing commands to generate design templates or boilerplate files based on predefined layouts or specifications.",
    "Asset Management: Running commands to manage design assets, such as organizing files, renaming, or tagging assets for easy retrieval.",
    "Color Palette Generation: Automating the generation of color palettes based on predefined rules, color theory principles, or sample images.",
    "Typography Management: Executing commands to manage typography settings, including font installation, text alignment, or kerning adjustments.",
    "Mockup Generation: Running commands to generate mockups or prototypes of design concepts, including layout variations, screen resolutions, or device orientations.",
    "Export Automation: Automating the export of design files to different formats or resolutions, including web-ready assets, print-ready files, or design comps.",
    "Image Editing: Executing commands to perform image editing tasks, such as cropping, rotating, or retouching images using design software or command-line tools.",
    "Data Visualization: Running commands to create visualizations of data sets, including charts, graphs, or infographics, based on predefined templates or configurations.",
    "UI/UX Testing: Automating user interface (UI) and user experience (UX) testing tasks, including generating test cases, capturing screenshots, or simulating user interactions.",
    "Design Collaboration: Executing commands to facilitate collaboration among team members, including sharing files, providing feedback, or syncing changes between collaborators.",
    "Design System Management: Running commands to manage design systems, including updating component libraries, documenting design guidelines, or publishing design assets.",
    "Animation Creation: Automating the creation of animations or motion graphics, including keyframe animation, tweening, or effects rendering.",
    "Print Production: Executing commands to prepare design files for print production, including color separation, layout imposition, or preflight checks.",
    "Design Automation Scripts: Running custom scripts or workflows to automate repetitive design tasks or integrate design processes with other tools or systems.",
    "Workflow Optimization: Automating workflow optimization tasks, such as streamlining file handoff, reducing manual steps, or improving collaboration efficiency among design team members.",
    "Social Media Posting: Automating the posting of content to social media platforms such as Facebook, Twitter, or Instagram by running commands to schedule posts or publish updates.",
    "Content Sharing: Executing commands to share content from external sources, such as articles, blog posts, or multimedia files, on social media channels.",
    "Data Retrieval: Running commands to retrieve social media data, including posts, comments, likes, and shares, for analysis or monitoring purposes.",
    "User Engagement Analysis: Automating the analysis of user engagement metrics on social media platforms, including likes, comments, shares, and follower growth.",
    "Sentiment Analysis: Executing commands to perform sentiment analysis on social media content, including identifying positive, negative, or neutral sentiment expressed by users.",
    "Influencer Identification: Running commands to identify and analyze social media influencers based on criteria such as follower count, engagement rate, and content relevance.",
    "Trend Monitoring: Automating the monitoring of social media trends, hashtags, and topics to identify emerging trends or viral content.",
    "Social Listening: Executing commands to listen to conversations and mentions related to specific keywords, brands, or topics across social media platforms.",
    "Community Management: Running commands to manage online communities and forums, including moderating discussions, responding to user queries, and resolving issues.",
    "Social Media Analytics: Automating the collection and analysis of social media analytics data, including reach, impressions, clicks, and conversions.",
    "Social Media Advertising: Executing commands to manage social media advertising campaigns, including budget allocation, targeting options, and ad creative optimization.",
    "Hashtag Analysis: Running commands to analyze the usage and popularity of hashtags on social media platforms, including tracking hashtag performance and effectiveness.",
    "Competitor Analysis: Automating the collection and analysis of data on competitors' social media activities, including content strategy, engagement metrics, and audience demographics.",
    "Brand Reputation Management: Executing commands to monitor and manage brand mentions, reviews, and sentiment on social media platforms to protect and enhance brand reputation.",
    "Social Media Integration: Running commands to integrate social media data and features into websites, applications, or other digital platforms via APIs or webhooks.",
    "Social Media Listening Tools Integration: Automating the integration of social media listening tools or platforms with internal systems for centralized monitoring and analysis.",
    "Social Media Campaign Tracking: Executing commands to track the performance of social media campaigns, including click-through rates, conversion rates, and return on investment (ROI).",
    "User Profile Management: Running commands to manage user profiles and accounts on social media platforms, including updating profile information, settings, and privacy preferences.",
    "Social Media Automation Tools Integration: Automating the integration of social media automation tools or platforms with other marketing, CRM, or analytics systems.",
    "Social Media Crisis Management: Executing commands to respond to and mitigate social media crises, including managing negative feedback, addressing complaints, and issuing public statements.",
    "Route Planning: Running commands to generate optimal routes for transportation vehicles based on factors such as distance, traffic conditions, and delivery schedules.",
    "Vehicle Tracking: Automating the tracking of vehicles in real-time using GPS technology, including monitoring vehicle locations, speeds, and routes.",
    "Fleet Management: Executing commands to manage and monitor a fleet of vehicles, including assigning vehicles to routes, tracking maintenance schedules, and optimizing vehicle usage.",
    "Delivery Scheduling: Running commands to schedule deliveries, including assigning delivery times, routes, and vehicles based on customer preferences and logistical constraints.",
    "Inventory Management: Automating inventory management tasks such as stock counting, replenishment, and allocation using barcode scanning or RFID technology.",
    "Warehouse Automation: Executing commands to automate warehouse operations, including goods receiving, storage, picking, packing, and shipping processes.",
    "Order Processing: Running commands to process customer orders, including order verification, picking, packing, and shipping preparation.",
    "Supply Chain Visibility: Automating the tracking and monitoring of goods throughout the supply chain, including inbound and outbound shipments, warehousing, and distribution.",
    "Shipping Documentation: Executing commands to generate shipping documents such as bills of lading, packing lists, and customs declarations for international shipments.",
    "Freight Rate Calculation: Running commands to calculate freight rates based on factors such as distance, weight, volume, and shipping mode (e.g., air, sea, road).",
    "Customs Clearance: Automating customs clearance processes for international shipments, including submitting import/export documentation and paying customs duties and taxes.",
    "Temperature Monitoring: Executing commands to monitor and control temperature-sensitive shipments, such as perishable goods or pharmaceuticals, during transportation and storage.",
    "Load Optimization: Running commands to optimize the loading of vehicles and containers, including maximizing load capacity, balancing weight distribution, and minimizing empty space.",
    "Driver Management: Automating driver scheduling, assignments, and performance monitoring, including tracking driving hours, rest periods, and compliance with regulations.",
    "Fuel Management: Executing commands to monitor fuel consumption, track fuel purchases, and optimize fuel usage to reduce costs and environmental impact.",
    "Risk Assessment: Running commands to assess and mitigate risks associated with transportation and logistics operations, including route hazards, weather conditions, and security threats.",
    "Customer Communication: Automating communication with customers regarding order status, delivery updates, and scheduling changes via email, SMS, or mobile apps.",
    "Incident Management: Executing commands to respond to and manage incidents such as vehicle breakdowns, accidents, or delays, including coordinating recovery and alternative solutions.",
    "Performance Analysis: Running commands to analyze key performance indicators (KPIs) such as on-time delivery rates, order accuracy, and inventory turnover to identify areas for improvement.",
    "Regulatory Compliance: Automating compliance with transportation regulations, safety standards, and environmental requirements imposed by government agencies and industry associations.",
    "Food Safety Inspections: Running commands to schedule and conduct food safety inspections at restaurants, food processing facilities, and retail establishments.",
    "Temperature Monitoring: Automating the monitoring of food storage temperatures, including refrigerators, freezers, and hot holding units, to ensure compliance with food safety regulations.",
    "Sanitation Audits: Executing commands to perform sanitation audits of food preparation areas, equipment, and utensils to prevent contamination and ensure hygiene standards.",
    "Food Recall Management: Running commands to manage food recalls, including identifying affected products, issuing recall notices, and tracking product returns and disposal.",
    "Allergen Control: Automating allergen control measures in food production and service, including labeling, segregation, and cleaning procedures to prevent cross-contamination.",
    "HACCP Implementation: Executing commands to implement Hazard Analysis and Critical Control Points (HACCP) plans, including identifying hazards, establishing critical control points, and monitoring procedures.",
    "Traceability Systems: Running commands to establish and maintain traceability systems for food products, including lot tracking, batch numbering, and product recall capabilities.",
    "Supplier Verification: Automating supplier verification processes to ensure that food suppliers meet safety and quality standards, including conducting audits and inspections.",
    "Food Labeling Compliance: Executing commands to ensure compliance with food labeling regulations, including ingredient listing, nutritional information, and allergen declarations.",
    "Pest Control Management: Running commands to manage pest control activities in food facilities, including inspections, treatments, and preventive measures.",
    "Training and Certification: Automating employee training and certification programs on food safety and hygiene practices, including scheduling, tracking, and reporting training activities.",
    "Water Quality Monitoring: Executing commands to monitor and maintain water quality standards for food production and processing, including testing for contaminants and disinfection procedures.",
    "Waste Management: Running commands to manage food waste and by-products generated during food production and service, including disposal, recycling, and composting.",
    "Cleaning and Disinfection: Automating cleaning and disinfection procedures for food contact surfaces, equipment, and utensils to prevent microbial contamination.",
    "Quality Control Testing: Executing commands to perform quality control tests on food products, including sensory evaluation, microbiological testing, and chemical analysis.",
    "Menu Development: Running commands to develop and update menus based on seasonal availability, customer preferences, and nutritional considerations.",
    "Compliance Reporting: Automating the generation of compliance reports for regulatory agencies and auditors, including documentation of inspections, corrective actions, and compliance status.",
    "Kitchen Management: Executing commands to manage kitchen operations, including recipe management, inventory control, and production scheduling.",
    "Food Safety Training Materials: Running commands to create and distribute food safety training materials, including videos, presentations, and interactive modules.",
    "Emergency Preparedness: Automating emergency preparedness plans for food facilities, including procedures for power outages, natural disasters, and foodborne illness outbreaks.",
    "Reservation Management: Running commands to manage hotel room reservations, including booking, modification, and cancellation of reservations.",
    "Check-In and Check-Out Automation: Automating the check-in and check-out processes for hotel guests, including generating key cards, processing payments, and updating room status.",
    "Room Allocation: Executing commands to assign rooms to guests based on availability, preferences, and special requests.",
    "Housekeeping Management: Running commands to manage housekeeping operations, including room cleaning schedules, task assignments, and inventory replenishment.",
    "Inventory Management: Automating inventory management tasks for hotel amenities such as toiletries, linens, and minibar items.",
    "Guest Feedback Collection: Executing commands to collect feedback from hotel guests via surveys, reviews, or feedback forms.",
    "Event Management: Running commands to manage events and conferences hosted at the hotel, including room setup, catering arrangements, and audiovisual equipment setup.",
    "Billing and Invoicing: Automating billing and invoicing processes for guest charges, including room rates, additional services, and taxes.",
    "Customer Relationship Management (CRM): Executing commands to manage guest profiles and interactions, including tracking preferences, past stays, and loyalty program memberships.",
    "Point-of-Sale (POS) Integration: Running commands to integrate hotel systems with POS systems for food and beverage sales, spa services, and other on-site purchases.",
    "Staff Scheduling: Automating staff scheduling and shift assignments for various departments such as front desk, housekeeping, and food service.",
    "Facility Maintenance: Executing commands to manage maintenance requests, repairs, and preventive maintenance tasks for hotel facilities and equipment.",
    "Concierge Services: Running commands to provide concierge services to guests, including restaurant reservations, transportation arrangements, and local attraction recommendations.",
    "Security Management: Automating security measures such as surveillance monitoring, access control, and emergency response procedures.",
    "Guest Communication: Executing commands to send automated messages to guests regarding reservation confirmations, room assignments, and special offers.",
    "Revenue Management: Running commands to analyze demand trends, adjust room rates dynamically, and optimize revenue across different distribution channels.",
    "Compliance Reporting: Automating the generation of compliance reports for regulatory agencies and industry standards, including safety inspections, hygiene audits, and environmental certifications.",
    "Staff Training and Development: Executing commands to manage training programs and certifications for hotel staff, including scheduling, tracking, and reporting training activities.",
    "Energy Management: Running commands to monitor and control energy consumption in hotel facilities, including HVAC systems, lighting, and water usage.",
    "Marketing Campaigns: Automating marketing campaigns to promote hotel services and special offers through email marketing, social media, and online advertising platforms.",
    "Web Server Installation: Running commands to install and configure web server software such as Apache, Nginx, or Microsoft IIS.",
    "Configuration Management: Automating the configuration of web server settings, including virtual hosts, SSL certificates, and security policies.",
    "Server Monitoring: Executing commands to monitor web server performance, including CPU usage, memory consumption, and request throughput.",
    "Log File Analysis: Running commands to analyze web server log files, including access logs, error logs, and security logs for troubleshooting and auditing purposes.",
    "Backup and Recovery: Automating the backup and recovery of web server configurations, website files, and databases to prevent data loss and ensure business continuity.",
    "Security Patching: Executing commands to apply security patches and updates to web server software and dependencies to mitigate vulnerabilities and protect against cyber threats.",
    "Load Balancing Configuration: Running commands to configure load balancing for distributing incoming web traffic across multiple server instances to improve scalability and reliability.",
    "Web Application Deployment: Automating the deployment of web applications, including code deployment, database migration, and environment setup for development, testing, and production environments.",
    "Content Management System (CMS) Installation: Executing commands to install and configure CMS platforms such as WordPress, Drupal, or Joomla for building and managing dynamic websites.",
    "Domain Name Configuration: Running commands to configure domain names, DNS records, and SSL certificates for securing and accessing websites over HTTPS.",
    "Database Integration: Automating the integration of web servers with database management systems such as MySQL, PostgreSQL, or MongoDB for storing and retrieving dynamic content.",
    "Web Server Hardening: Executing commands to implement security hardening measures such as firewall rules, access controls, and intrusion detection/prevention systems to protect web server infrastructure from cyber threats.",
    "Content Delivery Network (CDN) Integration: Running commands to integrate web servers with CDN services for caching and delivering static content closer to end-users to improve website performance and scalability.",
    "Web Application Firewall (WAF) Configuration: Automating the configuration of WAF rules and policies to filter and block malicious traffic, prevent SQL injection, XSS attacks, and other web application vulnerabilities.",
    "Reverse Proxy Configuration: Executing commands to configure reverse proxy servers such as Nginx or HAProxy to improve web server performance, handle SSL termination, and route traffic to backend web applications.",
    "Web Server Log Rotation: Running commands to rotate and archive web server log files periodically to prevent disk space exhaustion and ensure efficient log management.",
    "Website Performance Optimization: Automating tasks such as image optimization, minification of CSS and JavaScript files, and browser caching to improve website loading speed and user experience.",
    "SSL/TLS Certificate Management: Executing commands to manage SSL/TLS certificates, including certificate issuance, renewal, and installation for securing web server communications with HTTPS.",
    "Server-side Scripting Configuration: Running commands to configure server-side scripting languages such as PHP, Python, or Ruby for dynamic content generation and interaction with databases.",
    "Server Health Checks: Automating health checks and monitoring alerts for web server instances, including uptime monitoring, service availability, and response time measurements to ensure high availability and reliability of web services.",
    "Donation Processing: Running commands to process and record donations received from donors, including generating acknowledgment letters, updating donor databases, and issuing tax receipts.",
    "Volunteer Management: Automating volunteer recruitment, scheduling, and communication processes, including volunteer registration, assignment tracking, and volunteer appreciation efforts.",
    "Fundraising Campaigns: Executing commands to manage fundraising campaigns, including online fundraising platforms, peer-to-peer fundraising initiatives, and crowdfunding campaigns.",
    "Grant Management: Running commands to manage the grant application and reporting process, including tracking deadlines, preparing grant proposals, and submitting grant reports.",
    "Event Planning: Automating event planning tasks for fundraising events, community outreach events, and awareness campaigns, including event registration, ticketing, and attendee management.",
    "Member Engagement: Executing commands to engage members and supporters through newsletters, email updates, and social media outreach, including segmentation, personalization, and scheduling.",
    "Advocacy Campaigns: Running commands to support advocacy campaigns on social justice, human rights, and environmental issues, including grassroots organizing, mobilization efforts, and lobbying activities.",
    "Program Evaluation: Automating program evaluation and impact assessment processes to measure the effectiveness of non-profit programs and initiatives, including data collection, analysis, and reporting.",
    "Financial Management: Executing commands to manage non-profit finances, including budgeting, accounting, and financial reporting, using accounting software or financial management systems.",
    "Donor Stewardship: Running commands to steward donors and cultivate relationships through personalized communications, donor recognition programs, and donor appreciation events.",
    "Non-Profit Governance: Automating governance processes, including board meeting scheduling, agenda distribution, and meeting minutes recording, to support effective governance and compliance.",
    "Volunteer Training: Executing commands to deliver volunteer training materials and resources, including online training modules, training videos, and quizzes, to enhance volunteer skills and knowledge.",
    "Impact Reporting: Running commands to generate impact reports for donors, stakeholders, and the public, including visualizations, infographics, and storytelling narratives to showcase non-profit achievements.",
    "Donor Research: Automating donor research processes to identify potential major donors, corporate sponsors, and foundation partners, including prospect research, wealth screening, and donor profiling.",
    "Non-Profit Marketing: Executing commands to develop and execute marketing campaigns to raise awareness of non-profit causes, attract new supporters, and engage existing stakeholders.",
    "Database Management: Running commands to manage non-profit databases, including donor databases, volunteer databases, and program participant databases, to ensure data accuracy and integrity.",
    "Grassroots Organizing: Automating grassroots organizing efforts to mobilize supporters, recruit volunteers, and coordinate community-based actions and campaigns.",
    "Non-Profit Collaboration: Executing commands to facilitate collaboration and partnerships with other non-profits, government agencies, and community organizations to leverage resources and achieve common goals.",
    "Resource Allocation: Running commands to allocate resources such as staff time, funding, and equipment to non-profit programs and initiatives based on strategic priorities and organizational goals.",
    "Compliance Monitoring: Automating compliance monitoring processes to ensure adherence to regulatory requirements, reporting obligations, and non-profit governance standards.",
]


package = ["Flask", "FastAPI", "Django", "Tornado", "Twisted", "aiohttp"]


prompt_template = """\n### Instruction: Given a function, generate a program for "{}" which perform {}, using the given function in the ### Input.

### Input:
{}

### Requirements:
- Has at least 2 function.
- The code must be compilable code under 50 lines.
- Remove the code comments.
- Change the name of the function to match with the task.
- Using "{}" package
- The code must include the provided function
"""

tokenizer = llm.get_tokenizer()
prompts = []
for func in funcs:
    for task in tasks:
        task_name = task.split(":")[0].strip()
        task_des = task.split(":")[-1].strip()
        for pack in package:
            message_text = [
                {
                    "role": "system",
                    "content": "You are an AI assistant for a software engineer. Generate a code satisfying the requirements given a context input.",
                },
                {
                    "role": "user",
                    "content": prompt_template.format(task_name, task_des, func, pack),
                },
            ]
            text = tokenizer.apply_chat_template(message_text, tokenize=False)
            prompts.append(text)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)

pred = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    pred.append(generated_text)

df = pd.DataFrame({"id": range(len(pred)), "pred": pred})


def extract_substring_between_tags(text):
    # print(sample_text)
    start_tag = "```python"
    end_tag = "```"

    # Find the position of the start tag
    start_index = text.find(start_tag)
    # print(start_index)
    if start_index == -1:
        return "N/A"  # Start tag not found

    # Find the position of the end tag
    end_index = text.find(end_tag, start_index + len(start_tag))
    # print(end_index)
    if end_index == -1:
        return "N/A"  # End tag not found

    # Extract the substring between the tags
    substring = text[start_index + len(start_tag) : end_index]
    return substring.strip()


df["generated_code"] = df["pred"].apply(lambda x: extract_substring_between_tags(x))
df["extractable"] = df["generated_code"].apply(lambda x: x != "N/A")
df_ext = df.loc[df["extractable"] == True].copy()
df_noext = df.loc[df["extractable"] == False].copy()


def remove_dup(x):
    try:
        ast.parse(x)
        return 1
    except:
        return 0


df_noext["ast_compilable"] = df_noext["pred"].apply(lambda x: remove_dup(x))
df_noext = df_noext.drop(
    df_noext.loc[df_noext["ast_compilable"] == False].index, axis=0
)
df_noext["generated_code"] = df["pred"].apply(lambda x: x.strip())
df = pd.concat([df_ext, df_noext], axis=0).sample(frac=1.0).reset_index(drop=True)
df["contain_injection"] = df["generated_code"].apply(
    lambda x: ("injection" in x) or ("Injection" in x)
)
df = df.loc[df["contain_injection"] == False].copy().reset_index(drop=True)
df["id"] = range(df.shape[0])
df = df.drop(["pred", "extractable", "ast_compilable", "contain_injection"], axis=1)
df["generated_code"] = df["generated_code"].apply(
    lambda x: x.replace("async def ", "def ")
)


def count_functions(script):
    """
    Count the number of functions in a Python script.

    Args:
    - script (str): Python script as a string.

    Returns:
    - int: Number of functions in the script.
    """
    # Parse the script's AST
    try:
        tree = ast.parse(script)
    except:
        return -1

    # Initialize counter
    function_count = 0

    # Iterate through the nodes of the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Found a function definition
            function_count += 1

    return function_count


df["num_func"] = df["generated_code"].apply(lambda x: count_functions(x))
df["has_main"] = df["generated_code"].apply(lambda x: '__name__ == "__main__":' in x)

temp_df = df.loc[(df["has_main"] == False) & (df["num_func"] == 1)].copy()
no_temp_df = (
    df.iloc[[i for i in range(df.shape[0]) if i not in list(temp_df.index)]]
    .copy()
    .reset_index(drop=True)
)
temp_df = temp_df.reset_index(drop=True)
prompt_template = """\n### Instruction: Given a function, please perform the requirements 

### Input:
{}

### Requirements:
- Add "if __name__ == "__main__":" to the code and run the program in the __main__
"""

tokenizer = llm.get_tokenizer()
prompts = []
# print(len(prompts))
for func in tqdm(temp_df["generated_code"]):
    message_text = [
        {
            "role": "system",
            "content": "You are an AI assistant for a software engineer. Generate a code satisfying the requirements given a context input.",
        },
        {
            "role": "user",
            "content": prompt_template.format(func),
        },
    ]
    text = tokenizer.apply_chat_template(message_text, tokenize=False)
    # print(len(prompts))
    prompts.append(text)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)

pred = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    pred.append(generated_text)

temp_df["pred"] = pred
temp_df["generated_code"] = temp_df["pred"].apply(
    lambda x: extract_substring_between_tags(x)
)
temp_df["extractable"] = temp_df["generated_code"].apply(lambda x: x != "N/A")
df_ext = temp_df.loc[temp_df["extractable"] == True].copy()
df_noext = temp_df.loc[temp_df["extractable"] == False].copy()
df_noext["ast_compilable"] = df_noext["pred"].apply(lambda x: remove_dup(x))
df_noext["ast_compilable"].value_counts()
df_noext = df_noext.drop(
    df_noext.loc[df_noext["ast_compilable"] == False].index, axis=0
)
df_noext["generated_code"] = temp_df["pred"].apply(lambda x: x.strip())
temp_df = pd.concat([df_ext, df_noext], axis=0).sample(frac=1.0).reset_index(drop=True)
temp_df = temp_df[no_temp_df.columns]
df = pd.concat([no_temp_df, temp_df], axis=0).sort_values("id").reset_index(drop=True)
df["has_main"] = df["generated_code"].apply(lambda x: '__name__ == "__main__":' in x)
df = (
    df.loc[(df["num_func"] > 1) | (df["has_main"] == True)]
    .copy()
    .reset_index(drop=True)
)
df["id"] = range(df.shape[0])
path = []
name = []

code_path = "./gen_data/cwe-20/codes"
os.mkdirs(code_path)

for i in range(df.shape[0]):
    name.append(f"sample_{i}.py")
    path.append(os.path.join(code_path, f"sample_{i}.py"))
    with open(os.path.join(code_path, f"sample_{i}.py"), "w") as f:
        f.write(df.at[i, "generated_code"])

df["path"] = path
df["name"] = name

cmd = "codeql database create {} --language=python --overwrite --source-root {} --threads=32 && codeql database analyze {} $CODEQL_HOME/codeql-repo/python/ql/src/Security/CWE-0{}/ --format=csv --output={} --threads=32 --no-save-cache --ram=64000"
cmd = cmd.format(
    os.path.join("./", f"cqldb"),
    code_path,
    os.path.join("./", f"cqldb"),
    20,
    os.path.join("./", f"cqlres-cwe20.csv"),
)

p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
r = p.stdout.read().decode("utf-8") + p.stderr.read().decode("utf-8")

df_res = pd.read_csv("./gen_data/cwe-20/codeql-cwe20.csv", header=None)
df_res.columns = [f"Col_{i}" for i in range(df_res.shape[1])]
df_res["Col_4"] = df_res["Col_4"].apply(lambda x: x[1:])


def detect_scope(code, line_number):
    try:
        tree = ast.parse(code)
    except:
        return "N/A"
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.lineno <= line_number <= node.end_lineno:
                return f"Func-{node.name}-{node.lineno}-{node.end_lineno}"
        # elif isinstance(node, ast.ClassDef):
        #     if node.lineno <= line_number <= node.end_lineno:
        #         return f"Class-{node.name}-{node.lineno}-{node.end_lineno}"
    return f"Global-{line_number}"


name = []
vul_func = []
path = "./gen_data/cwe-20/codes/"
df_res = df_res.groupby("Col_4")["Col_5"].apply(list)
for key, item in zip(df_res.index, df_res):
    funcs = ""
    with open(os.path.join(path, key), "r") as f:
        codes = f.read()
        for index in item:
            funcs += f"|{detect_scope(code=codes, line_number=index)}"
    if key in name:
        idx = name.index(key)
        if len(funcs[1:]) > len(vul_func[idx]):
            vul_func[idx] = funcs[1:]
    else:
        name.append(key)
        vul_func.append(funcs[1:])

df["codeql-cwe20"] = 0
df["vul_func20_codeql"] = "N/A"

res_df = pd.DataFrame({"new_name": name, "vul_func20_codeql": vul_func})

res_df["codeql-cwe20"] = 1
res_df = res_df.reset_index(drop=True)
if res_df["new_name"].duplicated().sum() > 0:
    print(res_df.head())
update_df = df.loc[df["name"].isin(res_df["new_name"])].copy().reset_index(drop=True)
non_update_df = (
    df.loc[df["name"].isin(res_df["new_name"]) == False].copy().reset_index(drop=True)
)
update_df = update_df.drop(["codeql-cwe20", "vul_func20_codeql"], axis=1)
update_df = update_df.merge(res_df, left_on="name", right_on="new_name")
df = (
    pd.concat([non_update_df, update_df], axis=0)
    .sort_values("id")
    .reset_index(drop=True)
)
df = df.drop(["new_name", "has_main"], axis=1)
df.to_csv("../csv/generated-cwe20.csv", index=False)
