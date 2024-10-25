/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:48:16 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.ListIterator;
import java.util.Properties;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Parser_ESTest extends Parser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      PosixParser posixParser0 = new PosixParser();
      Properties properties0 = new Properties();
      // Undeclared exception!
      try { 
        posixParser0.parse(options0, stringArray0, properties0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.PosixParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-";
      CommandLine commandLine0 = basicParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "";
      stringArray0[1] = "-";
      Properties properties0 = new Properties();
      // Undeclared exception!
      try { 
        basicParser0.parse(options0, stringArray0, properties0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "*";
      stringArray0[1] = ",";
      stringArray0[2] = "-0";
      Properties properties0 = new Properties();
      try { 
        basicParser0.parse(options0, stringArray0, properties0, false);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -0
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[10];
      stringArray0[0] = "-%Wl6n<$3M]$N";
      Properties properties0 = new Properties();
      PosixParser posixParser0 = new PosixParser();
      CommandLine commandLine0 = posixParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      properties0.putIfAbsent("`=u8J6oV9\"Eh[JfG", "`=u8J6oV9\"Eh[JfG");
      // Undeclared exception!
      try { 
        basicParser0.parse(options0, (String[]) null, properties0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      Options options1 = options0.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      Properties properties0 = new Properties();
      try { 
        basicParser0.parse(options1, stringArray0, properties0, true);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required option: []
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      Options options1 = options0.addOptionGroup(optionGroup0);
      Options options2 = options1.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[4];
      stringArray0[0] = "";
      Properties properties0 = new Properties();
      try { 
        basicParser0.parse(options2, stringArray0, properties0, true);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required options: [][]
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = new Option("0", " *(~I3]t)", true, "]|9u.&uTogHNRrW.Jy");
      options0.addOption(option0);
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "*";
      stringArray0[1] = ",";
      stringArray0[2] = "-0";
      stringArray0[3] = "0";
      Properties properties0 = new Properties();
      CommandLine commandLine0 = basicParser0.parse(options0, stringArray0, properties0, false);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-%W%l6n<KM~]N";
      basicParser0.parse(options0, stringArray0, properties0, true);
      Option option0 = new Option((String) null, true, "-%W%l6n<KM~]N");
      option0.setOptionalArg(true);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((String) null).when(listIterator0).next();
      doReturn("-%W%l6n<KM~]N").when(listIterator0).previous();
      basicParser0.processArgs(option0, listIterator0);
      assertTrue(option0.hasArg());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "0";
      posixParser0.parse(options0, stringArray0);
      Option option0 = new Option("0", "0", true, "0");
      options0.addOption(option0);
      option0.setRequired(true);
      ListIterator<Object> listIterator0 = (ListIterator<Object>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true, true).when(listIterator0).hasNext();
      doReturn("0", "0").when(listIterator0).next();
      doReturn(option0).when(listIterator0).previous();
      posixParser0.processOption("0", listIterator0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = new Option("0", " *(~I3]t)", true, "]|9u.&uTogHNRrW.Jy");
      OptionGroup optionGroup0 = new OptionGroup();
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup1);
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "*";
      stringArray0[1] = ",";
      stringArray0[2] = "-0";
      Properties properties0 = new Properties();
      try { 
        basicParser0.parse(options0, stringArray0, properties0, false);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option:0
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PosixParser posixParser0 = new PosixParser();
      Options options0 = new Options();
      Option option0 = new Option("", ";+%)1 FDV\u0001BjNec}mU", true, "]|9u.&uTogHNRrW.Jy");
      OptionGroup optionGroup0 = new OptionGroup();
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      optionGroup1.setRequired(true);
      String[] stringArray0 = new String[23];
      stringArray0[0] = "";
      posixParser0.parse(options1, stringArray0, true);
      ListIterator<Object> listIterator0 = (ListIterator<Object>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true, true).when(listIterator0).hasNext();
      doReturn(";+%)1 FDV\u0001BjNec}mU", "]|9u.&uTogHNRrW.Jy").when(listIterator0).next();
      doReturn(option0).when(listIterator0).previous();
      posixParser0.processOption("", listIterator0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = new Option((String) null, (String) null, false, (String) null);
      Options options1 = options0.addOption(option0);
      PosixParser posixParser0 = new PosixParser();
      posixParser0.parse(options1, (String[]) null, false);
      posixParser0.processOption((String) null, (ListIterator) null);
  }
}
