/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:26:23 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.PrintWriter;
import java.util.Comparator;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HelpFormatter_ESTest extends HelpFormatter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getWidth();
      assertEquals(74, int0);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLongOptPrefix("usage: ");
      assertEquals("usage: ", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getDescPadding();
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, int0);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setNewLine("usage: ");
      assertEquals("usage: ", helpFormatter0.getNewLine());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLeftPadding(1);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setArgName("-");
      assertEquals("-", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setSyntaxPrefix("arg");
      assertEquals("arg", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp((-2792), "--", "--", (Options) null, "--");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      helpFormatter0.printHelp("-", options0);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setWidth(3);
      assertEquals(3, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      assertEquals("-", helpFormatter0.getOptPrefix());
      
      helpFormatter0.setOptPrefix("");
      assertEquals("", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getLeftPadding();
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getArgName();
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", string0);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp((PrintWriter) null, 1, "eGgW`148-b4C,<rSe3_", "z&", (Options) null, 1, 1, "arg");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Text too long for line - throwing exception to avoid infinite loop [CLI-162]:        eGgW`148-b4C,<rSe3_
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      assertEquals(3, helpFormatter0.defaultDescPad);
      
      helpFormatter0.setDescPadding(1);
      assertEquals(1, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp("%(#+;mQ2MV V%", "", (Options) null, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getNewLine();
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("\n", string0);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp("-", (Options) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getLongOptPrefix();
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", string0);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getSyntaxPrefix();
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("usage: ", string0);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getOptPrefix();
      assertEquals("-", string0);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Options options1 = options0.addOption("arg", "arg", true, "-");
      options1.addOption("qrCsK", "", true, "--");
      helpFormatter0.printHelp(74, "-", "--", options1, "--", true);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Comparator<String> comparator0 = (Comparator<String>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      helpFormatter0.setOptionComparator(comparator0);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptionComparator((Comparator) null);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp(74, (String) null, (String) null, (Options) null, "", true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cmdLineSyntax not provided
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp(3, "", "", options0, "", false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cmdLineSyntax not provided
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      helpFormatter0.printHelp(74, "8;=ze:mJF}G^*", "", options0, "", true);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", true, "arg");
      optionGroup0.addOption(option0);
      Option option1 = new Option("A", true, "\n");
      OptionGroup optionGroup1 = optionGroup0.addOption(option1);
      Options options1 = options0.addOptionGroup(optionGroup1);
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp(3, "\n", "--", options1, "--", true);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Text too long for line - throwing exception to avoid infinite loop [CLI-162]:        [- <arg> | -A <arg>]
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("tgXrA", true, "\n");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      optionGroup1.setRequired(true);
      Options options1 = options0.addOptionGroup(optionGroup1);
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp(3, "tgXrA", "--", options1, "--", true);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Text too long for line - throwing exception to avoid infinite loop [CLI-162]:        tgXrA -tgXrA <arg>
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption((String) null, "\n", true, "`[\"li:ywA>");
      helpFormatter0.printHelp(31, "`[\"li:ywA>", "`[\"li:ywA>", options0, "`[\"li:ywA>", true);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Options options1 = options0.addOption("arg", false, "arg");
      helpFormatter0.printHelp(74, "-", "--", options1, "--", true);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption("arg", true, "-");
      // Undeclared exception!
      try { 
        helpFormatter0.renderOptions((StringBuffer) null, 32, options0, 32, 32);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption("arg", (String) null, true, (String) null);
      helpFormatter0.printHelp(74, "-", "--", options0, "--", true);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim((String) null);
      assertNull(string0);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim("");
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("", string0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }
}