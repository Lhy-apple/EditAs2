/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:01:50 GMT 2023
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
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(74, int0);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLongOptPrefix("arg");
      assertEquals("arg", helpFormatter0.getLongOptPrefix());
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
      helpFormatter0.setNewLine("<Sj");
      assertEquals("<Sj", helpFormatter0.getNewLine());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLeftPadding(3);
      assertEquals(3, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setArgName("--");
      assertEquals("--", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setSyntaxPrefix("--");
      assertEquals("--", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      assertEquals(3, helpFormatter0.defaultDescPad);
      
      Options options0 = new Options();
      helpFormatter0.printHelp("arg", options0);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setWidth(74);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptPrefix("-");
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getLeftPadding();
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(1, int0);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getArgName();
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("arg", string0);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp((PrintWriter) null, 8, "", "", options0, 8, 8, "arg");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // cmdLineSyntax not provided
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setDescPadding(3);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp("T|xy/~Aua", "T|xy/~Aua", (Options) null, "T|xy/~Aua");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getNewLine();
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("\n", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp("pca)", (Options) null, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getLongOptPrefix();
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", string0);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getSyntaxPrefix();
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("usage: ", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getOptPrefix();
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("-", string0);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Comparator<Integer> comparator0 = (Comparator<Integer>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      helpFormatter0.setOptionComparator(comparator0);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptionComparator((Comparator) null);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp(74, (String) null, (String) null, options0, (String) null);
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
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("", "");
      optionGroup0.addOption(option0);
      Option option1 = new Option("arg", "arg", false, "~[!*E&~(P8TYpf9.");
      OptionGroup optionGroup1 = optionGroup0.addOption(option1);
      Options options1 = options0.addOptionGroup(optionGroup1);
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, 3, "~[!*E&~(P8TYpf9.", options1);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Total width is less than the width of the argument and indent - no room for the description
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("arg", "arg", true, "~[!*E&~(P8TYpf9.");
      optionGroup0.setRequired(true);
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      Options options1 = options0.addOptionGroup(optionGroup1);
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, 3, "~[!*E&~(P8TYpf9.", options1);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Total width is less than the width of the argument and indent - no room for the description
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption("", true, "--");
      helpFormatter0.printHelp((int) (byte)106, "\n", "\n", options0, " ", true);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption("", "usage: ", false, "usage: ");
      Options options1 = options0.addOption((String) null, "\n", false, "usage: ");
      helpFormatter0.printHelp((int) (byte)106, "\n", "\n", options1, " ", true);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Options options1 = options0.addOption("arg", true, "arg");
      StringBuffer stringBuffer0 = new StringBuffer((CharSequence) "--");
      // Undeclared exception!
      try { 
        helpFormatter0.renderOptions(stringBuffer0, 74, options1, 74, 74);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Total width is less than the width of the argument and indent - no room for the description
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption("arg", true, (String) null);
      helpFormatter0.printHelp(4094, "usage: ", "usage: ", options0, "usage: ");
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      helpFormatter0.printHelp(8, "arg", "arg", options0, "The addValue method is not intended for client use. Subclasses should use the addValueForProcessing method instead. ");
      assertEquals(1, helpFormatter0.getLeftPadding());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.findWrapPos("\n", (-1447), (-1447));
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim((String) null);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim("");
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("", string0);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim("\n");
      assertEquals("", string0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }
}