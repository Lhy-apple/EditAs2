/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:37:02 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class PosixParser_ESTest extends PosixParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[7];
      stringArray0[0] = "";
      stringArray0[1] = "";
      stringArray0[2] = "--=*n_LL4Y$Ms(i+QP2S,";
      CommandLine commandLine0 = posixParser0.parse(options0, stringArray0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      Option option0 = new Option("", "");
      Options options1 = options0.addOption(option0);
      String[] stringArray0 = new String[7];
      stringArray0[0] = "";
      stringArray0[1] = "";
      stringArray0[2] = "--=*n_LL4Y$Ms(i+QP2S,";
      // Undeclared exception!
      try { 
        posixParser0.parse(options1, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(1, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "--");
      optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[4];
      stringArray0[0] = "--";
      stringArray0[1] = "-org.apache.commons.cli.PosixParser";
      // Undeclared exception!
      try { 
        posixParser0.parse(options1, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[8];
      stringArray0[0] = "-";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(8, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "--");
      option0.setLongOpt("org.apache.commons.cli.PosixParser");
      optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[4];
      stringArray0[0] = "--";
      stringArray0[1] = "-org.apache.commons.cli.PosixParser";
      // Undeclared exception!
      try { 
        posixParser0.parse(options1, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "--");
      optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[7];
      stringArray0[0] = "--";
      stringArray0[1] = "--";
      stringArray0[2] = "";
      String[] stringArray1 = posixParser0.flatten(options1, stringArray0, true);
      assertEquals(8, stringArray1.length);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      options0.addOption("", false, "--Ewe");
      String[] stringArray0 = new String[8];
      stringArray0[0] = "-";
      // Undeclared exception!
      try { 
        posixParser0.flatten(options0, stringArray0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "--");
      optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[7];
      stringArray0[0] = "--";
      stringArray0[1] = "--";
      stringArray0[2] = "";
      stringArray0[3] = "--";
      stringArray0[4] = "--";
      stringArray0[5] = "";
      stringArray0[6] = "--";
      posixParser0.parse(options1, stringArray0);
      posixParser0.burstToken("--", true);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "--");
      optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup0);
      option0.setArgs((-2));
      String[] stringArray0 = new String[7];
      stringArray0[0] = "--";
      stringArray0[1] = "--";
      stringArray0[2] = "--";
      stringArray0[3] = "--";
      stringArray0[4] = "--";
      stringArray0[5] = "";
      stringArray0[6] = "--";
      posixParser0.parse(options1, stringArray0);
      posixParser0.burstToken("--", true);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "--");
      optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup0);
      option0.setArgs(2);
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      stringArray0[1] = "";
      stringArray0[2] = "";
      posixParser0.parse(options0, stringArray0);
      posixParser0.burstToken("--opt contains illegal character value '", true);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "-u[?K;='!HB2M!|xO,[";
      String[] stringArray1 = posixParser0.flatten(options0, stringArray0, true);
      assertEquals(5, stringArray1.length);
  }
}