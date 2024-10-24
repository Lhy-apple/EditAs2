/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:06:11 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.PrintWriter;
import java.util.Comparator;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionGroup;
import org.apache.commons.cli.Options;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockPrintWriter;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HelpFormatter_ESTest extends HelpFormatter_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getWidth();
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(74, int0);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
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
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, int0);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setNewLine("&iGxzQ6spn{ x");
      assertEquals("&iGxzQ6spn{ x", helpFormatter0.getNewLine());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLeftPadding(74);
      assertEquals(74, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      assertEquals("arg", helpFormatter0.getArgName());
      
      helpFormatter0.setArgName("");
      assertEquals("", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      helpFormatter0.printHelp("arg", "\n", options0, "-");
      assertEquals("--", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setSyntaxPrefix("usage: ");
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp(74, "", "d", options0, "d");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cmdLineSyntax not provided
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      helpFormatter0.printHelp(" ", options0);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setWidth(74);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptPrefix("--");
      assertEquals("--", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getLeftPadding();
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(1, int0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getArgName();
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("usage: ");
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp((PrintWriter) mockPrintWriter0, 32, (String) null, "", (Options) null, 32, 32, "");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cmdLineSyntax not provided
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      assertEquals(3, helpFormatter0.defaultDescPad);
      
      helpFormatter0.setDescPadding(1);
      assertEquals(1, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getNewLine();
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("\n", string0);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getLongOptPrefix();
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", string0);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getSyntaxPrefix();
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getOptPrefix();
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("-", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Comparator comparator0 = helpFormatter0.getOptionComparator();
      helpFormatter0.setOptionComparator(comparator0);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptionComparator((Comparator) null);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      assertEquals(74, helpFormatter0.defaultWidth);
      
      Options options0 = new Options();
      helpFormatter0.printHelp("usage: ", "n]F6", options0, "");
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Options options1 = options0.addOption((String) null, true, "usage: ");
      helpFormatter0.printHelp("arg", "\n", options1, (String) null, true);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Option option0 = new Option("", false, "");
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.addOption(option0);
      Options options0 = new Options();
      Option option1 = new Option(" ", "", true, "");
      OptionGroup optionGroup1 = optionGroup0.addOption(option1);
      options0.addOptionGroup(optionGroup1);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("ykfx,S$");
      helpFormatter0.printUsage((PrintWriter) mockPrintWriter0, 783, "k:Q[$dQAx", options0);
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("UJFsOb", true, "\n");
      optionGroup0.setRequired(true);
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup1);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("UJFsOb");
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) mockPrintWriter0, (-2), "arg", options0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("UJFsOb", true, "\n");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      option0.setArgName("");
      options0.addOptionGroup(optionGroup1);
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("UJFsOb");
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) mockPrintWriter0, (-2), "arg", options0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption("arg", "arg", true, "-");
      StringBuffer stringBuffer0 = new StringBuffer(1);
      helpFormatter0.renderOptions(stringBuffer0, 3, options0, 3, 3);
      assertEquals(37, stringBuffer0.length());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      StringBuffer stringBuffer0 = new StringBuffer();
      Options options0 = new Options();
      Option option0 = new Option("arg", false, "arg");
      Options options1 = options0.addOption(option0);
      option0.setArgName("");
      option0.setArgs(2545);
      helpFormatter0.renderOptions(stringBuffer0, 31, options1, 7, 31);
      assertEquals(129, stringBuffer0.length());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Options options1 = options0.addOption("", true, "arg");
      options1.addOption("arg", false, "arg");
      StringBuffer stringBuffer0 = new StringBuffer();
      helpFormatter0.renderOptions(stringBuffer0, 74, options0, 74, 919);
      assertEquals(517, stringBuffer0.length());
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      assertEquals(74, helpFormatter0.defaultWidth);
      
      Options options0 = new Options();
      Options options1 = options0.addOption("arg", true, (String) null);
      helpFormatter0.printHelp("--", options1, true);
      assertEquals("arg", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      StringBuffer stringBuffer0 = new StringBuffer();
      // Undeclared exception!
      try { 
        helpFormatter0.renderWrappedText(stringBuffer0, 3, (-8), "I!.Kx <m6x-0pG3[");
        fail("Expecting exception: NegativeArraySizeException");
      
      } catch(NegativeArraySizeException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.lang.AbstractStringBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      MockPrintWriter mockPrintWriter0 = new MockPrintWriter("arg");
      Options options0 = new Options();
      Option option0 = new Option("", "-");
      Options options1 = options0.addOption(option0);
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) mockPrintWriter0, 1, "f", options1);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      StringBuffer stringBuffer0 = new StringBuffer(3);
      helpFormatter0.renderWrappedText(stringBuffer0, 3, 74, "\n");
      assertEquals(3, stringBuffer0.length());
      assertEquals("\n  ", stringBuffer0.toString());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption("", true, "\n");
      StringBuffer stringBuffer0 = new StringBuffer(74);
      // Undeclared exception!
      try { 
        helpFormatter0.renderOptions(stringBuffer0, 0, options0, 0, 0);
        fail("Expecting exception: NegativeArraySizeException");
      
      } catch(NegativeArraySizeException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.lang.AbstractStringBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim((String) null);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      StringBuffer stringBuffer0 = new StringBuffer();
      StringBuffer stringBuffer1 = helpFormatter0.renderWrappedText(stringBuffer0, 3, 74, "");
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("", stringBuffer1.toString());
      assertEquals(74, helpFormatter0.defaultWidth);
  }
}
