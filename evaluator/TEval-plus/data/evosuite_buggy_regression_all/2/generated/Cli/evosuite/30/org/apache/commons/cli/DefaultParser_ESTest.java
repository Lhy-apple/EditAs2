/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:22:41 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
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
      CommandLine commandLine0 = defaultParser0.parse(options0, (String[]) null);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-.p\"O";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, properties0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      properties0.put("X-.jpO", "X-.jpO");
      // Undeclared exception!
      try { 
        defaultParser0.parse(options0, (String[]) null, properties0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[9];
      stringArray0[0] = "--1{<TGu{NK`";
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      Options options1 = options0.addOptionGroup(optionGroup0);
      try { 
        defaultParser0.parse(options1, stringArray0, true);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required option: []
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("", true, "");
      defaultParser0.parse(options1, (String[]) null, true);
      defaultParser0.handleConcatenatedOptions("--aU{pP");
      defaultParser0.handleConcatenatedOptions("--aU{pP");
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[9];
      stringArray0[0] = "--";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, false);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[9];
      stringArray0[0] = "-K=j\"=";
      try { 
        defaultParser0.parse(options0, stringArray0, false);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -K=j\"=
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "--K^\"=";
      try { 
        defaultParser0.parse(options0, stringArray0, false);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: --K^\"=
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-=";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[9];
      stringArray0[0] = "-K^\"=";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      defaultParser0.parse(options1, (String[]) null, false);
      try { 
        defaultParser0.handleConcatenatedOptions("--aU{pP");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: --aU{pP
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      optionGroup1.setRequired(true);
      defaultParser0.parse(options1, (String[]) null, false);
      try { 
        defaultParser0.handleConcatenatedOptions("--aU{pP");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: --aU{pP
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("true", "", true, "--");
      defaultParser0.parse(options1, (String[]) null, true);
      defaultParser0.handleConcatenatedOptions("--");
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("", "", false, "");
      DefaultParser defaultParser0 = new DefaultParser();
      defaultParser0.parse(options1, (String[]) null, true);
      defaultParser0.handleConcatenatedOptions("---.p\"O");
  }
}
