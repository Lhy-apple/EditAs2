/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:50:18 GMT 2023
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
      String[] stringArray0 = new String[0];
      defaultParser0.parse(options0, stringArray0, true);
      defaultParser0.handleConcatenatedOptions("--S");
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      CommandLine commandLine0 = defaultParser0.parse(options0, (String[]) null);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", true, "");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      String[] stringArray0 = new String[0];
      defaultParser0.parse(options1, stringArray0, true);
      defaultParser0.handleConcatenatedOptions("--S");
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, properties0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      properties0.setProperty("!", "!");
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
  public void test05()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      options0.addRequiredOption("ZmKnw", "){xeV0/H!R'rcS", true, "){xeV0/H!R'rcS");
      String[] stringArray0 = new String[0];
      try { 
        defaultParser0.parse(options0, stringArray0, properties0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required option: ZmKnw
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("", true, "--S");
      String[] stringArray0 = new String[0];
      defaultParser0.parse(options1, stringArray0, true);
      defaultParser0.handleConcatenatedOptions("--S");
      defaultParser0.handleConcatenatedOptions("--S");
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[5];
      stringArray0[0] = "--";
      CommandLine commandLine0 = defaultParser0.parse(options0, stringArray0, properties0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "--IvI]=+(?{Z;p/br";
      try { 
        defaultParser0.parse(options0, stringArray0, properties0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: --IvI]=+(?{Z;p/br
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-B#MW=@n{";
      try { 
        defaultParser0.parse(options0, stringArray0, properties0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -B#MW=@n{
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "----;Yvq,2]HF,R;8I_2d";
      try { 
        defaultParser0.parse(options0, stringArray0, properties0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: ----;Yvq,2]HF,R;8I_2d
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[18];
      stringArray0[0] = "-S";
      DefaultParser defaultParser0 = new DefaultParser();
      try { 
        defaultParser0.parse(options0, stringArray0, properties0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -S
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-!$~dOsa`!zfBR%\"9";
      try { 
        defaultParser0.parse(options0, stringArray0, properties0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -!$~dOsa`!zfBR%\"9
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "-!=~dOsa`!zfBa\"9";
      try { 
        defaultParser0.parse(options0, stringArray0, properties0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -!=~dOsa`!zfBa\"9
         //
         verifyException("org.apache.commons.cli.DefaultParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      Options options1 = options0.addOption("", "--S");
      String[] stringArray0 = new String[0];
      defaultParser0.parse(options1, stringArray0, true);
      defaultParser0.handleConcatenatedOptions("--S");
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", true, "");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      optionGroup1.setRequired(true);
      String[] stringArray0 = new String[0];
      defaultParser0.parse(options1, stringArray0, true);
      defaultParser0.handleConcatenatedOptions("--S");
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      options0.addOption("", "");
      String[] stringArray0 = new String[8];
      stringArray0[0] = "";
      defaultParser0.parse(options0, stringArray0, true);
      defaultParser0.handleConcatenatedOptions("--");
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      DefaultParser defaultParser0 = new DefaultParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[8];
      stringArray0[0] = "";
      options0.addOption("", "", true, "wD");
      defaultParser0.parse(options0, stringArray0, true);
      defaultParser0.handleConcatenatedOptions("--");
  }
}