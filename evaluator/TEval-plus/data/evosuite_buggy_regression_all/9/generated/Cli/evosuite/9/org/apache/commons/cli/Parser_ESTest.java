/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:34:54 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Properties;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Parser_ESTest extends Parser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      CommandLine commandLine0 = gnuParser0.parse(options0, (String[]) null);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      CommandLine commandLine0 = gnuParser0.parse(options0, (String[]) null, properties0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[1];
      options0.addOption(" ", " ", true, " ");
      Properties properties0 = new Properties();
      GnuParser gnuParser0 = new GnuParser();
      // Undeclared exception!
      try { 
        gnuParser0.parse(options0, stringArray0, properties0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.GnuParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "-";
      stringArray0[1] = "-";
      stringArray0[2] = "-";
      stringArray0[3] = "-";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-q-ZuAnl";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-q-ZuAnl";
      try { 
        gnuParser0.parse(options0, stringArray0, properties0, false);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -q-ZuAnl
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "y5L;s9j'G(:qg3(4t";
      stringArray0[1] = "-";
      Properties properties0 = new Properties();
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, properties0, false);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "--";
      stringArray0[1] = "--";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      properties0.put("-q-ZuAnl", gnuParser0);
      // Undeclared exception!
      try { 
        gnuParser0.parse(options0, (String[]) null, properties0, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Properties properties0 = new Properties();
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      options0.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[3];
      stringArray0[0] = "e#(";
      stringArray0[1] = "e#(";
      stringArray0[2] = ">(F:A`pHL@\"";
      try { 
        gnuParser0.parse(options0, stringArray0, properties0, true);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required option: []
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Properties properties0 = new Properties();
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      Options options1 = options0.addOptionGroup(optionGroup0);
      options1.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[3];
      stringArray0[0] = "e#(";
      stringArray0[1] = "e#(";
      stringArray0[2] = ">(F:A`pHL@\"";
      try { 
        gnuParser0.parse(options0, stringArray0, properties0, true);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required options: [][]
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[11];
      stringArray0[0] = " ";
      basicParser0.parse(options0, stringArray0, true);
      options0.addOption(" ", " ", true, " ");
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(listIterator0).hasNext();
      try { 
        basicParser0.processOption(" ", listIterator0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option: 
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[2];
      stringArray0[0] = " ";
      basicParser0.parse(options0, stringArray0, true);
      options0.addOption(" ", " ", true, " ");
      ListIterator<InputStream> listIterator0 = (ListIterator<InputStream>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      try { 
        basicParser0.processOption(" ", listIterator0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option: 
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Option option0 = new Option("", "");
      option0.setOptionalArg(true);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      ListIterator<Object> listIterator0 = linkedList0.listIterator();
      basicParser0.processArgs(option0, listIterator0);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[4];
      stringArray0[0] = " ";
      basicParser0.parse(options0, stringArray0, true);
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option(" ", stringArray0[1]);
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup1);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      basicParser0.processOption(" ", listIterator0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[4];
      stringArray0[0] = " ";
      basicParser0.parse(options0, stringArray0, true);
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("yes", " ");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      optionGroup0.setRequired(true);
      options0.addOptionGroup(optionGroup1);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      basicParser0.processOption("yes", listIterator0);
  }
}