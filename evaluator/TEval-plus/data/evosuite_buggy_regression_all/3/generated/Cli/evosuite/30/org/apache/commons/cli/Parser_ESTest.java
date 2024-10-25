/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:06:06 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Locale;
import java.util.Properties;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Parser_ESTest extends Parser_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, properties0, false);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Options options0 = new Options();
      Properties properties0 = new Properties();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("2", "2");
      optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup0);
      GnuParser gnuParser0 = new GnuParser();
      gnuParser0.parse(options1, (String[]) null, properties0, true);
      gnuParser0.processOption("2", (ListIterator) null);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      String[] stringArray0 = new String[6];
      stringArray0[0] = "-[ Options: [ short java.util.HashMap@0000000004 ] [ long {&P9%=[oCtion: $ &P9%  [ARG] :: 1 ]} ]";
      Properties properties0 = new Properties();
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      String[] stringArray0 = new String[17];
      stringArray0[0] = "-";
      stringArray0[1] = "-[ Options: [ short java.util.HashMap@0000000004 ] [ long {&P9%=[oCtion: $ &P9%  [ARG] :: 1 ]} ]";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      String[] stringArray0 = Locale.getISOCountries();
      Options options0 = new Options();
      Properties properties0 = new Properties();
      BasicParser basicParser0 = new BasicParser();
      CommandLine commandLine0 = basicParser0.parse(options0, stringArray0, properties0, true);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "-[ Options: [Oshort javamutil.HashMap@0000000004 ] [ long {&P9%=[oCtion: $ &P9%  [ARG] :: 1 ]} ]";
      try { 
        gnuParser0.parse(options0, stringArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Unrecognized option: -[ Options: [Oshort javamutil.HashMap@0000000004 ] [ long {&P9%=[oCtion: $ &P9%  [ARG] :: 1 ]} ]
         //
         verifyException("org.apache.commons.cli.Parser", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      String[] stringArray0 = new String[1];
      stringArray0[0] = "1";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0, false);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      String[] stringArray0 = new String[2];
      stringArray0[0] = "--";
      stringArray0[1] = "--";
      CommandLine commandLine0 = gnuParser0.parse(options0, stringArray0);
      assertNotNull(commandLine0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Options options0 = new Options();
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      properties0.put("-%Xt9#jAsqwo%xT|q1l=", gnuParser0);
      // Undeclared exception!
      try { 
        gnuParser0.parse(options0, (String[]) null, properties0);
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
      Properties properties0 = new Properties();
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.setRequired(true);
      options0.addOptionGroup(optionGroup0);
      GnuParser gnuParser0 = new GnuParser();
      try { 
        gnuParser0.parse(options0, (String[]) null, properties0, false);
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
      Options options0 = new Options();
      options0.addOption("", "", true, "");
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      gnuParser0.parse(options0, (String[]) null, properties0, true);
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add("");
      ListIterator<String> listIterator0 = linkedList0.listIterator();
      gnuParser0.processOption("", listIterator0);
      assertFalse(listIterator0.hasNext());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Options options0 = new Options();
      Options options1 = options0.addOption("", "", true, "");
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      gnuParser0.parse(options1, (String[]) null, properties0, true);
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add("MEvq#T`}");
      ListIterator<String> listIterator0 = linkedList0.listIterator();
      gnuParser0.processOption("", listIterator0);
      assertFalse(listIterator0.hasNext());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Options options0 = new Options();
      options0.addOption("", "", true, "");
      GnuParser gnuParser0 = new GnuParser();
      Properties properties0 = new Properties();
      gnuParser0.parse(options0, (String[]) null, properties0, true);
      LinkedList<String> linkedList0 = new LinkedList<String>();
      linkedList0.add("-");
      ListIterator<String> listIterator0 = linkedList0.listIterator();
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
  public void test13()  throws Throwable  {
      GnuParser gnuParser0 = new GnuParser();
      Option option0 = new Option("", false, "");
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      option0.setOptionalArg(true);
      ListIterator<Object> listIterator0 = linkedList0.listIterator();
      gnuParser0.processArgs(option0, listIterator0);
      assertFalse(option0.hasArg());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Options options0 = new Options();
      Properties properties0 = new Properties();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("2", "2");
      optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup0);
      optionGroup0.setRequired(true);
      GnuParser gnuParser0 = new GnuParser();
      gnuParser0.parse(options1, (String[]) null, properties0, true);
      gnuParser0.processOption("2", (ListIterator) null);
  }
}
