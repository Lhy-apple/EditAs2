/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:28:50 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.Reader;
import java.io.StringReader;
import java.util.Properties;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DefaultParser_ESTest extends DefaultParser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      // Undeclared exception!
      try { 
        defaultParser0.parse(options0, stringArray0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      Options options1 = options0.addRequiredOption("e", "15r4tB@rl`SX6F{P", true, "15r4tB@rl`SX6F{P");
      String[] stringArray0 = new String[4];
      stringArray0[0] = "--1";
      stringArray0[1] = "-=";
      try { 
        defaultParser0.parse(options1, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -=
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "", true, "");
      optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[0];
      defaultParser0.parse(options0, stringArray0, properties0, true);
      defaultParser0.handleConcatenatedOptions("[- ]");
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      StringReader stringReader0 = new StringReader("--!JHRc:85C j;~TR");
      Properties properties0 = new Properties();
      properties0.load((Reader) stringReader0);
      DefaultParser defaultParser0 = new DefaultParser();
      try { 
        defaultParser0.parse(options0, (String[]) null, properties0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Default option wasn't defined
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      options0.addRequiredOption("", "", true, "");
      try { 
        defaultParser0.parse(options0, (String[]) null);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required option: 
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      options0.addOption("", "", true, "");
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      defaultParser0.parse(options0, stringArray0, properties0, true);
      Option option0 = new Option("", "");
      defaultParser0.currentOption = option0;
      defaultParser0.handleConcatenatedOptions("--");
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      String[] stringArray0 = new String[1];
      options0.addRequiredOption("1", "1", true, "-1");
      stringArray0[0] = "-1";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option: 1
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      String[] stringArray0 = new String[19];
      stringArray0[0] = "--";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      options0.addRequiredOption("gi", "15r4tB@rl`XF{P", true, "15r4tB@rl`XF{P");
      String[] stringArray0 = new String[9];
      stringArray0[0] = "--1";
      stringArray0[1] = "-m";
      // Undeclared exception!
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-";
      // Undeclared exception!
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      options0.addRequiredOption("eP", "15r4tB@rl`SX6F{P", true, "15r4tB@rl`SX6F{P");
      String[] stringArray0 = new String[7];
      stringArray0[0] = "--1";
      stringArray0[1] = "15r4tB@rl`SX6F{P";
      // Undeclared exception!
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      options0.addRequiredOption("gi", "15r4tB@rl`XF{P", true, "15r4tB@rl`XF{P");
      String[] stringArray0 = new String[9];
      stringArray0[0] = "--1";
      stringArray0[1] = "-";
      // Undeclared exception!
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      defaultParser0.parse(options0, (String[]) null, properties0, true);
      defaultParser0.handleConcatenatedOptions("--");
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "--@pI,9}q~A)*K}";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: --@pI,9}q~A)*K}
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      Options options1 = options0.addRequiredOption("A1", "--=", true, "--=");
      options1.addOption("true", "-15", true, "A1");
      String[] stringArray0 = new String[1];
      stringArray0[0] = "--=";
      try { 
        defaultParser0.parse(options1, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Ambiguous option: '--'  (could be: '--=', '-15')
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      Options options1 = options0.addRequiredOption("", "15r4tB@rlYSX6F{", false, "15r4tB@rlYSX6F{");
      String[] stringArray0 = new String[4];
      stringArray0[0] = "--=";
      try { 
        defaultParser0.parse(options1, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: --=
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      options0.addRequiredOption("", "15r4tB@rl`SX6F{P", true, "15r4tB@rl`SX6F{P");
      String[] stringArray0 = new String[4];
      stringArray0[0] = "--=";
      // Undeclared exception!
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-'Ij40%";
      DefaultParser defaultParser0 = new DefaultParser();
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -'Ij40%
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-U,?-i=.X^m]z>8";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -U,?-i=.X^m]z>8
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Options options0 = new Options();
      DefaultParser defaultParser0 = new DefaultParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-|=!WC|^s`%";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -|=!WC|^s`%
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Options options0 = new Options();
      options0.addOption("c_", "'", true, "c_");
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-'Ij40%";
      DefaultParser defaultParser0 = new DefaultParser();
      // Undeclared exception!
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      options0.addOption("", "");
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      defaultParser0.parse(options0, stringArray0, properties0, true);
      defaultParser0.handleConcatenatedOptions("---_AGS_wA^L|ED");
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "", true, "");
      optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup0);
      optionGroup0.setRequired(true);
      String[] stringArray0 = new String[0];
      defaultParser0.parse(options0, stringArray0, properties0, true);
      defaultParser0.handleConcatenatedOptions("[- ]");
  }
}