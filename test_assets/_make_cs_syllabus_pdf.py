"""Throwaway helper: render a 10-page Class 10 Computer Science syllabus PDF."""

import math
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

CONTENT = """Class 10 Computer Science - Syllabus
Subject: Computer Science
Class: 10

Unit 1 Computer Fundamentals
1.1 Introduction to Computers
A computer is an electronic device that accepts data, processes it according to a set of
instructions and produces information as output. Its main characteristics are speed, accuracy,
diligence, versatility, storage capacity and automation.
The functional block diagram of a computer has five parts: input unit, central processing unit,
memory unit, output unit and the control flow that links them.
Computers are classified by size and purpose into supercomputers, mainframes, minicomputers,
microcomputers and embedded systems. Each class differs in processing power, cost and use.
Applications of computers include education, banking, healthcare, scientific research,
entertainment, government services, transportation and e-commerce.
A typical computer system combines hardware, software, data, users and procedures to complete
a task reliably and repeatedly.

1.2 Hardware and Software
Hardware refers to the physical components of a computer such as the motherboard, processor,
memory modules, storage drives, keyboard, mouse and monitor.
Software is a set of programs and instructions that tell the hardware what to do. It is not
physical and is stored as files on a storage device.
System software manages the hardware and provides a platform for other programs. Examples are
operating systems, device drivers and utility programs.
Application software helps the user perform specific tasks, such as word processors,
spreadsheets, browsers, media players and games.
Utility software performs maintenance tasks like antivirus scanning, disk cleanup, file
compression and backup. Firmware is software stored in read-only memory that controls a device
at start-up.

1.3 Input and Output Devices
Input devices accept data and commands from the user and convert them into a form the computer
can process. Examples include the keyboard, mouse, joystick, scanner, microphone, webcam,
barcode reader and touchscreen.
Output devices present processed results to the user. Common examples are monitors, printers,
plotters, speakers, headphones and projectors.
Monitors may be LCD, LED or OLED and are described by resolution, refresh rate and size.
Printers are classified as impact (dot matrix) and non-impact (inkjet, laser).
A good input or output device should be accurate, fast, reliable, easy to use and compatible
with the rest of the computer system.

1.4 Memory and Storage
Memory is the part of a computer that holds data and instructions for processing. Primary
memory is directly accessed by the CPU and includes RAM, ROM, cache and registers.
RAM is volatile: its contents are lost when power is switched off. ROM is non-volatile and
stores firmware such as the BIOS that starts the computer.
Cache memory is a small, very fast memory placed between the CPU and RAM to reduce the
average time taken to access data.
Secondary storage holds data permanently. Examples are hard disk drives, solid state drives,
optical discs, USB flash drives, memory cards and cloud storage.
Memory units increase in steps of 1024: bit, byte, kilobyte, megabyte, gigabyte, terabyte and
petabyte. The memory hierarchy trades speed against cost and capacity.

Unit 2 Number Systems and Data Representation
2.1 Number Systems
A number system is defined by its base, or radix, which is the number of distinct digits it
uses. Digits in a number carry a place value that depends on their position.
The decimal system uses base 10 and digits 0 to 9. The binary system uses base 2 and digits
0 and 1, which suits the two states of an electronic switch.
The octal system uses base 8 and the hexadecimal system uses base 16 with digits 0 to 9 and
letters A to F. Hexadecimal is widely used as a short form of binary.
Numbers are converted between systems by repeated division for the integer part and repeated
multiplication for the fractional part.
Binary arithmetic follows simple rules for addition, subtraction and multiplication, and is
the basis of all calculations performed by a computer.

2.2 Computer Arithmetic and Codes
Signed numbers are represented using sign-magnitude, one's complement and two's complement
methods. Two's complement is preferred because it makes subtraction easy.
Binary coded decimal represents each decimal digit by a four-bit binary code.
Character codes map characters to numbers. ASCII uses seven or eight bits, while Unicode uses
a larger range to cover the scripts of the world.
Error detection and correction use extra bits such as parity bits, checksums and cyclic
redundancy checks to detect mistakes in transmission and storage.

2.3 Data Representation
Integers, fractions and negative numbers are stored in binary, using fixed or floating point
formats. Floating point representation stores a sign, an exponent and a mantissa.
Images are stored as a grid of picture elements called pixels. Each pixel holds colour
information, and resolution and colour depth decide image quality and file size.
Audio is stored by sampling a continuous sound wave at regular intervals and quantising each
sample. Video is a sequence of image frames played fast enough to appear continuous.
Compression reduces file size. Lossless compression keeps all data, while lossy compression
removes detail that the human eye or ear is unlikely to notice.

Unit 3 Computer Architecture
3.1 CPU and Its Components
The central processing unit executes instructions and controls the rest of the computer. It
contains the arithmetic logic unit, the control unit and a set of registers.
The arithmetic logic unit performs arithmetic and logical operations, while the control unit
decodes instructions and generates the timing and control signals.
A program is executed through the fetch, decode, execute and store cycle, repeated billions of
times per second. The clock speed measures how many cycles occur each second.
Instruction set architecture defines the instructions a processor supports. RISC designs use a
small set of simple instructions, while CISC designs use many complex instructions.

3.2 Motherboard and Buses
The motherboard is the main circuit board that connects the processor, memory, storage,
expansion cards and input and output ports.
A bus is a set of wires that carries data, addresses and control signals between components.
The data bus carries information, the address bus carries locations and the control bus
carries timing and command signals.
Chipsets manage communication between the processor, memory and peripheral devices.
Interrupts allow devices to signal the processor when they need attention.

3.3 Instruction Execution and Pipelining
Registers are small, very fast storage locations inside the CPU that hold operands, addresses
and intermediate results during execution.
Pipelining overlaps the stages of several instructions so that more than one instruction is in
progress at the same time, increasing throughput.
Common hazards in pipelines are data hazards, control hazards and structural hazards, which
are handled by forwarding, stalling and branch prediction.
Multi-core processors place several execution units on one chip, allowing true parallel
execution of independent threads.

Unit 4 Operating Systems
4.1 Introduction to Operating Systems
An operating system is system software that manages hardware and provides services to
application programs and users.
Its main functions are process management, memory management, file management, device
management and user interface provision.
Operating systems are classified as batch, multiprogramming, time-sharing, real-time,
distributed and network systems.
Popular examples include Microsoft Windows, Linux distributions, macOS, Android and iOS. A
Command Line Interface and a Graphical User Interface are two common ways of interacting with
an operating system.

4.2 File Systems and Management
A file is a named collection of related data stored on a storage device. A directory is a
container that organises files and other directories.
A path describes the location of a file in the directory tree, and may be absolute or
relative. Extensions such as .txt and .png indicate the file type.
Common file systems include FAT32, NTFS and ext4. They differ in the maximum file size,
security features and reliability.
File operations include creating, opening, reading, writing, renaming, copying, moving and
deleting files. Permissions control who may read, write or execute a file.

4.3 Processes and Scheduling
A process is a program in execution. It passes through states such as new, ready, running,
waiting and terminated, which are recorded in a process control block.
Scheduling decides which ready process uses the CPU next. Common algorithms are first come
first served, shortest job first, priority and round robin.
Multitasking allows several processes to share the CPU, while concurrency and parallelism
describe tasks that appear to run together or truly run at the same time.
Deadlock occurs when processes wait for one another forever. It can be prevented by avoiding
circular waits or detected and recovered by the operating system.

Unit 5 Application Software and Productivity Tools
5.1 Word Processing
A word processor is used to create, edit, format, save and print text documents.
Formatting includes fonts, sizes, colours, alignment, spacing, bullets, numbering and styles.
Styles and templates help keep long documents consistent and easy to update.
Tables organise information, while images, charts, headers, footers and page numbers improve
presentation. Mail merge produces many personalised letters from one template and a data
source.

5.2 Spreadsheets
A spreadsheet organises data in a grid of rows and columns. A cell is the intersection of a
row and a column, and a range is a group of cells.
Formulas begin with an equals sign and may contain functions. Common functions are SUM,
AVERAGE, MIN, MAX, COUNT, IF and VLOOKUP.
Relative and absolute references control how formulas change when they are copied to other
cells.
Charts turn numbers into graphs, and tools such as sorting, filtering, conditional formatting
and data validation help analyse and present data clearly.

5.3 Presentation Software
Presentation software creates slide shows for talks and demonstrations. Slides contain text,
images, charts, audio and video.
Layouts, themes, transitions and animations make a presentation attractive and readable.
Good presentations use one idea per slide, readable fonts, consistent colours and a logical
flow.
Speaker notes help the presenter, and exporting to PDF or video allows the presentation to be
shared widely.

5.4 Introduction to Database Tools
A database stores related data in an organised manner so it can be retrieved quickly. Data is
held in tables made of records and fields.
Each field stores one attribute of the entity, and each record stores one instance of it. A
primary key uniquely identifies a record.
Data types such as text, number, date and boolean decide what values a field may hold.
Queries, forms and reports in a database tool help enter, search and present data.

Unit 6 Programming Fundamentals
6.1 Introduction to Programming
An algorithm is a finite, ordered set of clear steps that solves a problem. It can be shown as
a flowchart using standard symbols or written as pseudocode.
Programming is the process of translating an algorithm into a language a computer can execute.
The program development cycle includes problem analysis, design, coding, testing, debugging
and maintenance.
Machine language, assembly language and high-level languages differ in readability, portability
and the amount of machine control they give.
A compiler translates the whole program at once, while an interpreter translates and runs it
line by line.

6.2 Python Basics
Python is a high-level, interpreted, general purpose language known for simple, readable
syntax.
Variables store values in named locations. Python supports integers, floats, strings, booleans,
lists, tuples, sets and dictionaries.
Input and output use the input and print functions. Comments begin with the hash symbol and
are ignored by the interpreter.
Operators include arithmetic, relational, logical and assignment operators, and expressions
are evaluated using the rules of precedence.
Type conversion changes a value from one data type to another using functions such as int,
float and str.

6.3 Control Structures
Selection uses if, elif and else to choose between different paths depending on a condition.
Indentation shows which statements belong to which block.
Iteration repeats statements. A for loop repeats a known number of times, while a while loop
repeats as long as a condition remains true.
The break statement exits a loop early, continue skips to the next repetition and else runs
after a loop ends without a break.
Nested loops and conditional statements allow patterns, tables and simple menus to be
generated.

6.4 Functions and Modules
A function is a named block of code that performs a task and can be called whenever needed.
It may accept parameters and return a value.
Defining a function avoids repetition, improves readability and makes programs easier to test
and maintain.
The scope of a variable decides where it can be used. Local variables exist inside a function,
while global variables exist throughout the program.
A module is a file of Python code that can be imported. The standard library provides many
ready modules for mathematics, random numbers, dates and file handling.

Unit 7 Data Structures and Algorithms
7.1 Basic Data Structures
A data structure organises data so it can be used efficiently. Lists store ordered, changeable
collections, while tuples store ordered, unchangeable collections.
Sets store unique, unordered elements, and dictionaries store key and value pairs for fast
lookup.
Strings are sequences of characters that support slicing, concatenation and useful methods
such as upper, lower, split and replace.
A stack follows last in first out order and is used for undo operations and expression
evaluation. A queue follows first in first out order and is used for scheduling and buffers.

7.2 Searching and Sorting
Linear search checks each element in turn and works on unsorted data, but becomes slow for
large lists.
Binary search repeatedly halves a sorted list and is much faster, but it requires the data to
be sorted first.
Sorting arranges data in a chosen order. Bubble sort, selection sort and insertion sort are
simple but slow on large lists.
More efficient algorithms such as merge sort and quick sort divide the problem into smaller
parts and combine the results.

7.3 Algorithm Design
Time complexity estimates how running time grows with input size, and space complexity does
the same for memory. Big O notation expresses this growth.
Recursion is a technique in which a function calls itself to solve a smaller version of a
problem, and must include a base case to stop.
Problem decomposition breaks a large task into smaller, manageable parts, and stepwise
refinement develops each part in greater detail.

Unit 8 Databases and SQL
8.1 Database Concepts
Data are raw facts, and information is processed data that is useful for decision making. A
database stores related data in an organised way.
A database management system is software that creates, controls and maintains databases, and
provides security, backup and query facilities.
The relational model stores data in tables called relations. A row is a record or tuple and a
column is a field or attribute.
A primary key uniquely identifies each record, and a foreign key links a record to another
table. Relationships may be one to one, one to many or many to many.

8.2 SQL Fundamentals
Structured Query Language is used to define, query and modify data in relational databases.
Data definition statements include CREATE, ALTER and DROP.
Data manipulation statements include INSERT, UPDATE and DELETE, and are used to change stored
data.
The SELECT statement retrieves data. Clauses such as WHERE, ORDER BY, GROUP BY and HAVING
filter, sort and summarise results.
Joins combine rows from two or more tables using a related column, and aggregate functions
such as COUNT, SUM, AVG, MIN and MAX summarise data.

8.3 Database Design and Security
Normalisation organises tables to reduce redundancy and avoid update, insert and delete
anomalies. It is carried out in stages called normal forms.
A transaction is a group of operations that must all succeed or all fail. Its ACID properties
are atomicity, consistency, isolation and durability.
Backups protect data against loss, and access control limits what each user may see or change.
Encryption and auditing protect privacy and help trace misuse of data.

Unit 9 Computer Networks and the Internet
9.1 Networking Basics
A computer network connects two or more devices so they can share data, hardware and
services.
Networks are classified by size as personal, local, metropolitan and wide area networks. The
internet is the largest network of networks.
Network topology describes how devices are arranged. Common topologies are star, bus, ring,
mesh, tree and hybrid, each with different cost and reliability.
Wired media such as twisted pair, coaxial and fibre optic cables differ in speed and distance.
Wireless media use radio waves, microwaves and infrared.

9.2 The Internet and the Web
Every device on a network has an address. An IP address identifies a device, and a domain name
system translates readable names into IP addresses.
A uniform resource locator gives the address of a resource on the web, made of a protocol, a
domain name and a path.
The web uses HTTP and HTTPS to transfer pages, and browsers display them. Servers, clients and
protocols work together to deliver content.
Services on the internet include email, file transfer, video conferencing, cloud storage,
social media and the internet of things.

9.3 Network Devices and Protocols
Hubs, switches, routers, modems, gateways and access points connect and direct traffic between
devices and networks.
A protocol is a set of rules for communication. The TCP/IP suite is the foundation of the
internet, and the OSI model describes networking in seven layers.
Bandwidth is the amount of data a link can carry per second, and latency is the delay before
data begins to arrive. Together they determine network performance.
Security on a network uses authentication, encryption and firewalls to protect data in
transit.

Unit 10 Cybersecurity, Ethics and Emerging Technologies
10.1 Cybersecurity Basics
Cybersecurity protects computers, networks and data from attack, damage and unauthorised
access.
Malware includes viruses, worms, trojans, spyware and ransomware. Each spreads and harms a
system in a different way.
Phishing and social engineering trick people into revealing passwords, bank details or
personal information.
Strong, unique passwords, two-factor authentication, regular updates, firewalls and antivirus
software greatly reduce risk.
Safe browsing habits include avoiding unknown links and downloads, checking website addresses
and keeping personal data private.

10.2 Digital Ethics and Law
Digital citizenship is the responsible and respectful use of technology.
Intellectual property includes copyright, patents and trademarks. Plagiarism is presenting
another person's work as one's own.
Software licences, such as proprietary and open source licences, state how a program may be
used, copied or modified.
Cyberbullying, spreading false information and invading privacy are harmful and may be
unlawful. Data protection laws and the Information Technology Act define rights and penalties.

10.3 Emerging Technologies
Artificial intelligence builds systems that perform tasks requiring human-like intelligence,
and machine learning learns patterns from data.
Robotics combines mechanics, electronics and software to build machines that sense and act in
the physical world.
Cloud computing provides computing services over the internet, while big data deals with very
large and varied data sets that require special tools.
Blockchain is a shared, tamper resistant record of transactions, and the internet of things
connects everyday devices to the network, raising new opportunities and new risks.

10.4 Practical Work and Assessment
Practical work includes using productivity tools, writing and running Python programs,
creating and querying simple databases, and using the internet safely.
Projects encourage problem solving, teamwork and creativity, and may combine several units of
the syllabus.
Assessment is continuous and final. It includes theory examinations, practical examinations,
project work and periodic tests.
The theory paper tests concepts, applications and reasoning, while the practical paper tests
hands-on skills in programming, databases and tools.
"""


def main() -> None:
    raw_lines = CONTENT.strip("\n").split("\n")
    lines: list[str] = []
    for line in raw_lines:
        if not line.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(line, width=88) or [""])

    pages = 10
    per_page = math.ceil(len(lines) / pages)

    target = Path("cs_syllabus.pdf")
    width, height = A4
    pdf = canvas.Canvas(str(target), pagesize=A4)
    pdf.setTitle("Class 10 Computer Science Syllabus")
    pdf.setSubject("Computer Science")
    pdf.setAuthor("AI Evaluation Helper")

    left, top, leading = 54, 60, 15
    y = height - top
    count = 0
    pdf.setFont("Helvetica", 10)
    for line in lines:
        if count >= per_page:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - top
            count = 0
        pdf.drawString(left, y, line)
        y -= leading
        count += 1
    pdf.save()

    print(f"wrote {target} | lines={len(lines)} | per_page={per_page}")


if __name__ == "__main__":
    main()
