/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:05:25 GMT 2023
 */

package org.apache.commons.cli;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.cli.Option;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Option_ESTest extends Option_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.getDescription();
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Option option0 = new Option("", "", false, "");
      option0.getArgName();
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.setOptionalArg(true);
      boolean boolean0 = option0.acceptsArg();
      assertTrue(option0.hasOptionalArg());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.setRequired(true);
      assertTrue(option0.isRequired());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Option option0 = new Option("", "", false, "");
      option0.isRequired();
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Option option0 = new Option("", "M^0*e$fo:(D[5;{", false, "");
      option0.getValuesList();
      assertEquals((-1), option0.getArgs());
      assertEquals("", option0.getDescription());
      assertEquals("", option0.getOpt());
      assertEquals("M^0*e$fo:(D[5;{", option0.getLongOpt());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Option option0 = new Option("", false, "");
      String string0 = option0.getLongOpt();
      assertEquals((-1), option0.getArgs());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      // Undeclared exception!
      try { 
        option0.getId();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli.Option", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      // Undeclared exception!
      try { 
        option0.addValue((String) null);
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // The addValue method is not intended for client use. Subclasses should use the addValueForProcessing method instead. 
         //
         verifyException("org.apache.commons.cli.Option", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, false, (String) null);
      Option option1 = (Option)option0.clone();
      boolean boolean0 = option0.equals(option1);
      assertEquals((-1), option1.getArgs());
      assertNotSame(option1, option0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, false, (String) null);
      option0.getType();
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Option option0 = new Option("NO_ARGS_ALLOWED", true, "NO_ARGS_ALLOWED");
      option0.setDescription("NO_ARGS_ALLOWED");
      assertFalse(option0.hasLongOpt());
      assertTrue(option0.hasArg());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.getOpt();
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, false, (String) null);
      option0.setLongOpt(" ");
      Option option1 = (Option)option0.clone();
      boolean boolean0 = option0.equals(option1);
      assertEquals((-1), option1.getArgs());
      assertTrue(boolean0);
      assertNotSame(option1, option0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Option option0 = new Option("", false, (String) null);
      int int0 = option0.getArgs();
      assertEquals((-1), int0);
      assertFalse(option0.hasLongOpt());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Option option0 = new Option("NO_ARGS_ALLOWED", true, "NO_ARGS_ALLOWED");
      option0.clearValues();
      assertEquals(1, option0.getArgs());
      assertFalse(option0.hasLongOpt());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Option option0 = new Option("", "", false, "");
      option0.getKey();
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Option option0 = new Option("", "");
      boolean boolean0 = option0.hasLongOpt();
      assertFalse(boolean0);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Option option0 = new Option("", "", false, "");
      boolean boolean0 = option0.hasLongOpt();
      assertTrue(boolean0);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Option option0 = new Option((String) null, false, "]1");
      boolean boolean0 = option0.hasArgName();
      assertEquals((-1), option0.getArgs());
      assertFalse(option0.hasLongOpt());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.setArgName(")+AM1KFG.pDSa");
      boolean boolean0 = option0.hasArgName();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Option option0 = new Option("", "");
      option0.setArgName("");
      boolean boolean0 = option0.hasArgName();
      assertEquals((-1), option0.getArgs());
      assertFalse(option0.hasLongOpt());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.setArgs(953);
      option0.toString();
      assertEquals(953, option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Option option0 = new Option("q", "q", true, "q");
      option0.setArgs((-2));
      option0.toString();
      assertEquals((-2), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Option option0 = new Option("NO_ARGS_ALLOWED", true, "NO_ARGS_ALLOWED");
      assertTrue(option0.hasArg());
      
      option0.addValueForProcessing("[ option: NO_ARGS_ALLOWED  [ARG] :: NO_ARGS_ALLOWED ]");
      String string0 = option0.getValue(" :: ");
      assertFalse(option0.hasLongOpt());
      assertEquals("[ option: NO_ARGS_ALLOWED  [ARG] :: NO_ARGS_ALLOWED ]", string0);
      assertFalse(option0.hasValueSeparator());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      // Undeclared exception!
      try { 
        option0.addValueForProcessing((String) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // NO_ARGS_ALLOWED
         //
         verifyException("org.apache.commons.cli.Option", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Option option0 = new Option("NO_ARGS_ALLOWED", true, "NO_ARGS_ALLOWED");
      option0.setValueSeparator('_');
      option0.addValueForProcessing("[ option: NO_ARGS_ALLOWED  [ARG] :: NO_ARGS_ALLOWED ]");
      assertEquals('_', option0.getValueSeparator());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Option option0 = new Option("NO_ARGS_ALLOWED", true, "NO_ARGS_ALLOWED");
      option0.setValueSeparator('_');
      option0.addValueForProcessing(" :: ");
      // Undeclared exception!
      try { 
        option0.addValueForProcessing("[ option: NO_ARGS_ALLOWED  [ARG] :: NO_ARGS_ALLOWED ]");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot add value, list full.
         //
         verifyException("org.apache.commons.cli.Option", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, false, (String) null);
      option0.getValue((String) null);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Option option0 = new Option("", "", true, "");
      option0.addValueForProcessing("");
      try { 
        option0.getValue((-2));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.getValue(44);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.setArgs(45);
      option0.addValueForProcessing("u@P");
      option0.getValues();
      assertEquals(45, option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      String[] stringArray0 = option0.getValues();
      assertEquals((-1), option0.getArgs());
      assertNull(stringArray0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Option option0 = new Option("NO_ARGS_ALLOWED", true, "NO_ARGS_ALLOWED");
      String string0 = option0.toString();
      assertEquals("[ option: NO_ARGS_ALLOWED  [ARG] :: NO_ARGS_ALLOWED ]", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Option option0 = new Option("", "");
      String string0 = option0.toString();
      assertEquals("[ option:   ::  ]", string0);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Option option0 = new Option("", "");
      Class<Object> class0 = Object.class;
      option0.setType(class0);
      String string0 = option0.toString();
      assertEquals((-1), option0.getArgs());
      assertEquals("[ option:   ::  :: class java.lang.Object ]", string0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Option option0 = new Option("", "", true, "");
      Option option1 = new Option("", true, "");
      boolean boolean0 = option0.equals(option1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Option option0 = new Option("", true, "8|z,");
      boolean boolean0 = option0.equals(option0);
      assertFalse(option0.hasLongOpt());
      assertEquals("", option0.getOpt());
      assertEquals("8|z,", option0.getDescription());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      boolean boolean0 = option0.equals((Object) null);
      assertFalse(boolean0);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      boolean boolean0 = option0.equals("");
      assertEquals((-1), option0.getArgs());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Option option0 = new Option("", "-8Mk[58");
      Option option1 = new Option("i", "", true, "");
      boolean boolean0 = option0.equals(option1);
      assertFalse(boolean0);
      assertFalse(option0.hasLongOpt());
      assertEquals((-1), option0.getArgs());
      assertEquals("i", option1.getOpt());
      assertEquals("", option1.getDescription());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      Option option1 = new Option("GkzuzvW", false, "GkzuzvW");
      boolean boolean0 = option0.equals(option1);
      assertFalse(boolean0);
      assertEquals((-1), option1.getArgs());
      assertFalse(option1.hasLongOpt());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Option option0 = new Option("", "");
      Option option1 = new Option("", "NO_ARGS_ALLOWED", false, " ");
      boolean boolean0 = option0.equals(option1);
      assertFalse(boolean0);
      assertEquals((-1), option1.getArgs());
      assertEquals("NO_ARGS_ALLOWED", option1.getLongOpt());
      assertEquals("", option1.getOpt());
      assertEquals(" ", option1.getDescription());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Option option0 = new Option((String) null, false, (String) null);
      option0.hashCode();
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Option option0 = new Option("", "", false, "C3(Oe<OoxrB/?vf&CU");
      option0.hashCode();
      assertEquals("", option0.getOpt());
      assertEquals((-1), option0.getArgs());
      assertEquals("C3(Oe<OoxrB/?vf&CU", option0.getDescription());
      assertEquals("", option0.getLongOpt());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      boolean boolean0 = option0.requiresArg();
      assertEquals((-1), option0.getArgs());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Option option0 = new Option((String) null, false, (String) null);
      option0.setOptionalArg(true);
      boolean boolean0 = option0.requiresArg();
      assertTrue(option0.hasOptionalArg());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Option option0 = new Option("", "", false, "");
      assertFalse(option0.hasArgs());
      
      option0.setArgs((-2));
      boolean boolean0 = option0.requiresArg();
      assertEquals((-2), option0.getArgs());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Option option0 = new Option((String) null, false, "]1");
      assertEquals((-1), option0.getArgs());
      
      option0.setArgs((-2));
      assertEquals((-2), option0.getArgs());
      
      option0.addValueForProcessing((String) null);
      boolean boolean0 = option0.requiresArg();
      assertFalse(boolean0);
  }
}
