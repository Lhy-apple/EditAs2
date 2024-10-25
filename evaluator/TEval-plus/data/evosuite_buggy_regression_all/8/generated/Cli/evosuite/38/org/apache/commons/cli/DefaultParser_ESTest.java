/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:01:35 GMT 2023
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
      options0.addRequiredOption("y", "y", true, "-y");
      String[] stringArray0 = new String[9];
      stringArray0[0] = "y";
      stringArray0[1] = "y";
      stringArray0[2] = "y";
      stringArray0[3] = "y";
      stringArray0[4] = "-y";
      stringArray0[5] = "--7=3eWF";
      stringArray0[6] = "-y(es";
      stringArray0[7] = "-y";
      stringArray0[8] = "-y";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option: y
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("y", true, "y");
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-y";
      stringArray0[1] = "y";
      CommandLine commandLine0 = defaultParser0.parse(options1, stringArray0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-true";
      Properties properties0 = new Properties();
      Options options1 = options0.addOption("true", "true");
      CommandLine commandLine0 = defaultParser0.parse(options1, stringArray0, properties0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      StringReader stringReader0 = new StringReader("_%}6H5{");
      properties0.load((Reader) stringReader0);
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
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addRequiredOption("y", "y", true, "y");
      String[] stringArray0 = new String[0];
      try { 
        defaultParser0.parse(options1, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required option: y
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "");
      optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup0);
      defaultParser0.parse(options0, (String[]) null);
      defaultParser0.currentOption = option0;
      defaultParser0.handleConcatenatedOptions("--");
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[19];
      stringArray0[0] = "--hS,O6zY";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "--";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("y", true, "y");
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-y";
      stringArray0[1] = "-y";
      try { 
        defaultParser0.parse(options1, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option: y
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "--V_{FiBC.`=+M0h>73";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: --V_{FiBC.`=+M0h>73
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("y", "y", true, "y");
      String[] stringArray0 = new String[2];
      stringArray0[0] = "--y";
      // Undeclared exception!
      try { 
        defaultParser0.parse(options1, stringArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-t";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -t
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-QYW=#DHw";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -QYW=#DHw
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-t=3eWF";
      try { 
        defaultParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -t=3eWF
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      optionGroup1.setRequired(true);
      defaultParser0.parse(options1, (String[]) null);
      try { 
        defaultParser0.handleConcatenatedOptions("--tYtrue");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: --tYtrue
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("", "", true, "");
      defaultParser0.parse(options1, (String[]) null, true);
      defaultParser0.handleConcatenatedOptions("--");
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      defaultParser0.parse(options0, (String[]) null);
      defaultParser0.stopAtNonOption = true;
      defaultParser0.handleConcatenatedOptions("--true");
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      options0.addOption("", "", false, "");
      defaultParser0.parse(options0, (String[]) null);
      defaultParser0.stopAtNonOption = true;
      defaultParser0.handleConcatenatedOptions("--true");
  }
}
