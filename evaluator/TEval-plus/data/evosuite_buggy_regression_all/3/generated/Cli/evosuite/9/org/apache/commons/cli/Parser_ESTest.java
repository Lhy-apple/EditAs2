/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:04:04 GMT 2023
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
import org.apache.commons.cli.PosixParser;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Parser_ESTest extends Parser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      OptionGroup optionGroup0 = new OptionGroup();
      Options options0 = new Options();
      optionGroup0.setRequired(true);
      Options options1 = options0.addOptionGroup(optionGroup0);
      options1.addOptionGroup(optionGroup0);
      String[] stringArray0 = new String[0];
      try { 
        gnuParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing required options: [][]
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Options options0 = new Options();
      PosixParser posixParser0 = new PosixParser();
      CommandLine commandLine0 = posixParser0.parse(options0, (String[]) null, false);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("", true, "");
      GnuParser gnuParser0 = new GnuParser();
      gnuParser0.parse(options1, (String[]) null);
      ListIterator<Integer> listIterator0 = (ListIterator<Integer>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      try { 
        gnuParser0.processOption("", listIterator0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option:
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-v;}lVF4";
      PosixParser posixParser0 = new PosixParser();
      CommandLine commandLine0 = posixParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      CommandLine commandLine0 = basicParser0.parse(options0, stringArray0, properties0, false);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "-v;=l?VF4";
      CommandLine commandLine0 = basicParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[12];
      stringArray0[0] = "-v;=?F4";
      try { 
        basicParser0.parse(options0, stringArray0, properties0, false);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -v;=?F4
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[4];
      stringArray0[0] = "v;=l?VF4";
      CommandLine commandLine0 = basicParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Options options0 = new Options();
      BasicParser basicParser0 = new BasicParser();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "-";
      Properties properties0 = new Properties();
      properties0.setProperty("-", "-");
      // Undeclared exception!
      try { 
        basicParser0.parse(options0, stringArray0, properties0, true);
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
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      options0.addOptionGroup(optionGroup0);
      GnuParser gnuParser0 = new GnuParser();
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
  public void test10()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Option option0 = new Option("", "");
      LinkedList<InputStream> linkedList0 = new LinkedList<InputStream>();
      ListIterator<InputStream> listIterator0 = linkedList0.listIterator();
      try { 
        gnuParser0.processArgs(option0, listIterator0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Missing argument for option:
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = new Option("MBA", "MBA", true, "MBA");
      BasicParser basicParser0 = new BasicParser();
      Properties properties0 = new Properties();
      Options options1 = options0.addOption("MBA", true, "MBA");
      String[] stringArray0 = new String[1];
      stringArray0[0] = "MBA";
      basicParser0.parse(options1, stringArray0, properties0);
      ListIterator<Object> listIterator0 = (ListIterator<Object>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true, false).when(listIterator0).hasNext();
      doReturn("MBA", (Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      basicParser0.processArgs(option0, listIterator0);
      assertEquals((-1), Option.UNINITIALIZED);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Option option0 = new Option("", "");
      GnuParser gnuParser0 = new GnuParser();
      option0.setOptionalArg(true);
      ListIterator<Object> listIterator0 = (ListIterator<Object>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(listIterator0).hasNext();
      gnuParser0.processArgs(option0, listIterator0);
      assertNull(option0.getValue());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("U", "U");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      GnuParser gnuParser0 = new GnuParser();
      gnuParser0.parse(options1, (String[]) null);
      gnuParser0.processOption("U", (ListIterator) null);
  }
}
