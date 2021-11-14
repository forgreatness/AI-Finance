﻿using System;

namespace interview_prep
{
    class Program
    {
        static void Main(string[] args)
        {
            var account = new BankAccount("Danh Nguyen", 1000);
            account.MakeWithdrawal(500, DateTime.Now, "Rent payment");
            Console.WriteLine(account.Balance);
            account.MakeDeposit(100, DateTime.Now, "Friend paid me back");
            Console.WriteLine(account.Balance);

            try {
                var invalidAccount = new BankAccount("invalid", -55);
            } catch (ArgumentOutOfRangeException e) {
                Console.WriteLine("Exception caught creating account with negative balance");
                Console.WriteLine(e.ToString());
            }


            // Test for a negative balance.
            try {
                account.MakeWithdrawal(750, DateTime.Now, "Attempt to overdraw");
            }
            catch (InvalidOperationException e) {
                Console.WriteLine("Exception caught trying to overdraw");
                Console.WriteLine(e.ToString());
            }

            Console.WriteLine(account.GetAccountHistory());
        }
    }
}
