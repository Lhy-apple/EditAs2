/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:03:56 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Properties;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
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
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[0];
      Properties properties0 = new Properties();
      CommandLine commandLine0 = basicParser0.parse(options0, stringArray0, properties0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Options options0 = new Options();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-ChsKg'Wr";
      PosixParser posixParser0 = new PosixParser();
      Properties properties0 = new Properties();
      CommandLine commandLine0 = posixParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[3];
      stringArray0[0] = "-";
      stringArray0[1] = "";
      stringArray0[2] = "";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[8];
      stringArray0[0] = "-";
      Properties properties0 = new Properties();
      CommandLine commandLine0 = basicParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-Ih?-'}xtsXb\"<";
      try { 
        gnuParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -Ih?-'}xtsXb\"<
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "Ca7p";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      String[] stringArray0 = new String[0];
      gnuParser0.parse(options0, stringArray0, false);
      Properties properties0 = new Properties();
      properties0.put("~Kbk ", "~Kbk ");
      // Undeclared exception!
      try { 
        gnuParser0.processProperties(properties0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      options0.addOptionGroup(optionGroup0);
      try { 
        gnuParser0.parse(options0, (String[]) null);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required option: []
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      Options options1 = options0.addOptionGroup(optionGroup0);
      Options options2 = options1.addOptionGroup(optionGroup0);
      try { 
        gnuParser0.parse(options2, (String[]) null);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required options: [], []
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[0];
      Properties properties0 = new Properties();
      basicParser0.parse(options0, stringArray0, properties0, true);
      Option option0 = new Option((String) null, "kQki<FrKrj*");
      ListIterator<Integer> listIterator0 = (ListIterator<Integer>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      try { 
        basicParser0.processArgs(option0, listIterator0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option:null
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Option option0 = new Option("", "", true, "");
      option0.addValueForProcessing("");
      ListIterator<Integer> listIterator0 = (ListIterator<Integer>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(listIterator0).hasNext();
      gnuParser0.processArgs(option0, listIterator0);
      assertFalse(option0.hasArgs());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      ListIterator<Object> listIterator0 = linkedList0.listIterator();
      Option option0 = new Option((String) null, "iPQ#/:BVgb");
      option0.setOptionalArg(true);
      basicParser0.processArgs(option0, listIterator0);
      assertEquals("arg", option0.getArgName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("tr_u", "an option from this group has already been selected: '");
      optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup0);
      Properties properties0 = new Properties();
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[1];
      option0.setRequired(true);
      stringArray0[0] = "-Ih?-w}xteXb\"<";
      basicParser0.parse(options0, stringArray0, properties0, true);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      basicParser0.processOption("tr_u", listIterator0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      Option option0 = new Option((String) null, (String) null);
      Options options1 = options0.addOption(option0);
      String[] stringArray0 = new String[0];
      gnuParser0.parse(options1, stringArray0);
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      ListIterator<Object> listIterator0 = linkedList0.listIterator();
      gnuParser0.processOption((String) null, listIterator0);
      assertFalse(listIterator0.hasPrevious());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("tr_u", "-Ih?-w}xteXb\"<");
      optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup0);
      optionGroup0.setRequired(true);
      Properties properties0 = new Properties();
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-Ih?-w}xteXb\"<";
      basicParser0.parse(options0, stringArray0, properties0, true);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      basicParser0.processOption("tr_u", listIterator0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BasicParser basicParser0 = new BasicParser();
      Options options0 = new Options();
      String[] stringArray0 = new String[0];
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      ListIterator<Object> listIterator0 = linkedList0.listIterator();
      Properties properties0 = new Properties();
      Options options1 = options0.addOption((String) null, (String) null, true, (String) null);
      basicParser0.parse(options1, stringArray0, properties0, false);
      try { 
        basicParser0.processOption((String) null, listIterator0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option:null
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }
}
