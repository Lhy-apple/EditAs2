/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 12:26:42 GMT 2023
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
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HelpFormatter_ESTest extends HelpFormatter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("arg");
      Options options0 = new Options();
      helpFormatter0.printHelp((PrintWriter) mockPrintWriter0, 74, "usage: ", "usage: ", options0, 1, 3, "usage: ", true);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getWidth();
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(74, int0);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLongOptPrefix("usage: ");
      assertEquals("usage: ", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getDescPadding();
      assertEquals(3, int0);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setNewLine("\n");
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("\n", helpFormatter0.getNewLine());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLeftPadding(74);
      assertEquals(74, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setArgName("usage: ");
      assertEquals("usage: ", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setSyntaxPrefix("--");
      assertEquals("--", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setWidth((-896));
      assertEquals((-896), helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptPrefix("--");
      assertEquals("--", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getLeftPadding();
      assertEquals(1, int0);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getArgName();
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", string0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("--");
      Options options0 = new Options();
      helpFormatter0.printHelp((PrintWriter) mockPrintWriter0, 1, "\n", "-", options0, 74, 1, (String) null);
      assertEquals(3, HelpFormatter.DEFAULT_DESC_PAD);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setDescPadding(54);
      assertEquals(54, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      helpFormatter0.printHelp("\"v2'{JR2K|il", "\"v2'{JR2K|il", options0, "");
      assertEquals("--", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getNewLine();
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("\n", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getLongOptPrefix();
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", string0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getSyntaxPrefix();
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("usage: ", string0);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getOptPrefix();
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", string0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("9>xwY?XU-");
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "", true, "arg");
      optionGroup0.addOption(option0);
      Options options0 = new Options();
      Option option1 = new Option("4NWZX", "9>xwY?XU-");
      optionGroup0.addOption(option1);
      Options options1 = options0.addOptionGroup(optionGroup0);
      helpFormatter0.printUsage((PrintWriter) mockPrintWriter0, 3, "4NWZX", options1);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Comparator<String> comparator0 = (Comparator<String>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      helpFormatter0.setOptionComparator(comparator0);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptionComparator((Comparator) null);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp((String) null, (String) null, options0, "-");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cmdLineSyntax not provided
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp("", options0, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cmdLineSyntax not provided
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp(59, "usage: ", "", (Options) null, "usage: ");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Options options1 = options0.addOption("arg", false, "LU=|80G");
      helpFormatter0.printHelp(" :: ", options1, true);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("arg");
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("arg", "The option '", true, "usage: ");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      optionGroup0.setRequired(true);
      helpFormatter0.printUsage((PrintWriter) mockPrintWriter0, 3, "arg", options1);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption((String) null, "?", true, "NCAey~A;f(");
      helpFormatter0.printHelp("\n", options0, true);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Option option0 = new Option("L", true, "?");
      Options options1 = options0.addOption(option0);
      option0.setArgName("");
      helpFormatter0.printHelp("\n", options1, true);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "", true, "arg");
      optionGroup0.addOption(option0);
      Options options0 = new Options();
      Option option1 = new Option("4NWZX", "9>xwY?XU-");
      optionGroup0.addOption(option1);
      Options options1 = options0.addOptionGroup(optionGroup0);
      helpFormatter0.printHelp("arg", options1, false);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Option option0 = new Option("arg", "A CloneNotSupportedException was thrown: ", false, "--");
      Options options1 = options0.addOption(option0);
      helpFormatter0.printHelp("A CloneNotSupportedException was thrown: ", options1, false);
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Options options1 = options0.addOption((String) null, false, (String) null);
      helpFormatter0.printHelp(helpFormatter0.DEFAULT_LONG_OPT_PREFIX, options1);
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.findWrapPos("\n", (-1), 1);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(0, int0);
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim((String) null);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("arg", helpFormatter0.getArgName());
      assertNull(string0);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim("");
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("", string0);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.renderWrappedText((StringBuffer) null, 4, 4, " ");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }
}