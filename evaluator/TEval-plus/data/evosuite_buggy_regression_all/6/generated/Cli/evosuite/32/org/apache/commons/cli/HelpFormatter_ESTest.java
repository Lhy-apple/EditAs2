/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:51:38 GMT 2023
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
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, int0);
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLongOptPrefix(" ");
      assertEquals(" ", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getDescPadding();
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(3, int0);
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLongOptSeparator("oRC3?I");
      assertEquals("oRC3?I", helpFormatter0.getLongOptSeparator());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setNewLine("YO1'?");
      assertEquals("YO1'?", helpFormatter0.getNewLine());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setLeftPadding((-332));
      assertEquals((-332), helpFormatter0.defaultLeftPad);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setArgName("Pbb2AGCj");
      assertEquals("Pbb2AGCj", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setSyntaxPrefix(", ");
      assertEquals(", ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      helpFormatter0.printHelp(74, " ", " ", options0, " ");
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals(3, helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      helpFormatter0.printHelp("arg", options0);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getLongOptSeparator();
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(" ", string0);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setWidth((-3311));
      assertEquals((-3311), helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptPrefix("' was specified but an option from this group ");
      assertEquals("' was specified but an option from this group ", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      int int0 = helpFormatter0.getLeftPadding();
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, int0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp((PrintWriter) null, 74, "oa$dSm", "oa$dSm", options0, 22, 0, "oa$dSm");
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
      helpFormatter0.setDescPadding((-425));
      assertEquals((-425), helpFormatter0.defaultDescPad);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getNewLine();
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("\n", string0);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp("' was specified but an option from this group ", (Options) null, false);
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
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("--", string0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getSyntaxPrefix();
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals("usage: ", string0);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.getOptPrefix();
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals("-", string0);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals("--", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Option option0 = new Option("arg", "arg", true, "usage: ");
      Options options1 = options0.addOption(option0);
      options1.addOption("Fv$mlS", true, "\"");
      helpFormatter0.printHelp("YmM!hrpr:S'7|kYS4h1", "YmM!hrpr:S'7|kYS4h1", options1, "YmM!hrpr:S'7|kYS4h1");
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Comparator<Object> comparator0 = (Comparator<Object>) mock(Comparator.class, new ViolatedAssumptionAnswer());
      helpFormatter0.setOptionComparator(comparator0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      helpFormatter0.setOptionComparator((Comparator) null);
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
      assertEquals("-", helpFormatter0.getOptPrefix());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      // Undeclared exception!
      try { 
        helpFormatter0.printHelp((String) null, "[~o]]77)\"rRaeC", options0, (String) null);
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
        helpFormatter0.printHelp("", "", options0, "");
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
      helpFormatter0.printHelp("Zzm50+!`2:vooFLz", "Zzm50+!`2:vooFLz", options0, "Zzm50+!`2:vooFLz", true);
      assertEquals(1, HelpFormatter.DEFAULT_LEFT_PAD);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Option option0 = new Option("", true, "Zzm50+[!`2:votk#.FLz");
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.addOption(option0);
      Options options0 = new Options();
      Options options1 = options0.addOption("Kki", true, "Zzm50+[!`2:votk#.FLz");
      options1.addOptionGroup(optionGroup0);
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, 64, "`mu9:c(mzI%", options0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("arg", true, "--");
      optionGroup0.setRequired(true);
      optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup0);
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, (-1), (String) null, options0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Options options0 = new Options();
      Option option0 = new Option((String) null, true, "-");
      Options options1 = options0.addOption(option0);
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, (-1), "-", options1);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Option option0 = new Option("", "usage: ");
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup0);
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, (-1), "", options0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      OptionGroup optionGroup0 = new OptionGroup();
      Option option0 = new Option("arg", true, "--");
      option0.setArgName("--");
      OptionGroup optionGroup1 = optionGroup0.addOption(option0);
      options0.addOptionGroup(optionGroup1);
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, (-1067), "_~", options0);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Option option0 = new Option("", true, "Zzm50+[!`2:votk#.FLz");
      option0.setArgName("");
      OptionGroup optionGroup0 = new OptionGroup();
      optionGroup0.addOption(option0);
      Options options0 = new Options();
      options0.addOptionGroup(optionGroup0);
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, 64, "`mu9:c(mzI%", options0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption((String) null, " ", true, (String) null);
      helpFormatter0.printHelp(" ", "=$v*-HeT#lStHCmj", options0, "8]l");
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      Option option0 = new Option((String) null, "--");
      options0.addOption(option0);
      helpFormatter0.printHelp("-", "-", options0, "-");
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(74, helpFormatter0.defaultWidth);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      Options options0 = new Options();
      options0.addOption("arg", "usage: ", true, "-");
      StringBuffer stringBuffer0 = new StringBuffer();
      // Undeclared exception!
      try { 
        helpFormatter0.renderOptions(stringBuffer0, 1, options0, 2474, 1);
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      StringBuffer stringBuffer0 = new StringBuffer();
      // Undeclared exception!
      try { 
        helpFormatter0.renderWrappedText(stringBuffer0, 3, 31, "The addValue method is not intended for client use. Subclasses should use the addValueForProcessing method instead. ");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      StringBuffer stringBuffer0 = new StringBuffer();
      // Undeclared exception!
      try { 
        helpFormatter0.renderWrappedText(stringBuffer0, (-1679), 1, "\n");
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      try { 
        helpFormatter0.printUsage((PrintWriter) null, 74, "\n");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.HelpFormatter", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      String string0 = helpFormatter0.rtrim((String) null);
      assertEquals("arg", helpFormatter0.getArgName());
      assertEquals(" ", helpFormatter0.getLongOptSeparator());
      assertEquals(3, helpFormatter0.defaultDescPad);
      assertEquals(1, helpFormatter0.defaultLeftPad);
      assertEquals("-", helpFormatter0.getOptPrefix());
      assertEquals("usage: ", helpFormatter0.getSyntaxPrefix());
      assertNull(string0);
      assertEquals(74, helpFormatter0.defaultWidth);
      assertEquals("--", helpFormatter0.getLongOptPrefix());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      // Undeclared exception!
      helpFormatter0.printUsage((PrintWriter) null, 0, "usage: ");
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HelpFormatter helpFormatter0 = new HelpFormatter();
      StringBuffer stringBuffer0 = new StringBuffer("--");
      helpFormatter0.renderWrappedText(stringBuffer0, 1, 1, " ");
  }
}
