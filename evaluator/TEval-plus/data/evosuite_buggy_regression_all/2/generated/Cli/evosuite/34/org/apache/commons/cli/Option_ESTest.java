/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:22:53 GMT 2023
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
      Option option0 = new Option("", "");
      option0.getArgName();
      assertFalse(option0.hasLongOpt());
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
      Option option0 = new Option("", "", false, "");
      option0.setRequired(false);
      assertFalse(option0.isRequired());
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Option option0 = new Option("I5", "");
      option0.isRequired();
      assertEquals("", option0.getDescription());
      assertFalse(option0.hasLongOpt());
      assertEquals("I5", option0.getOpt());
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Option option0 = new Option("", "");
      option0.getValuesList();
      assertEquals((-1), option0.getArgs());
      assertFalse(option0.hasLongOpt());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Option option0 = new Option("", "");
      String string0 = option0.getLongOpt();
      assertEquals((-1), option0.getArgs());
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Option option0 = new Option("", true, "");
      // Undeclared exception!
      try { 
        option0.getId();
        fail("Expecting exception: StringIndexOutOfBoundsException");
      
      } catch(StringIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Option option0 = new Option("", "");
      // Undeclared exception!
      try { 
        option0.addValue("");
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
      Option option0 = new Option((String) null, (String) null, false, "");
      option0.getType();
      assertEquals((-1), option0.getArgs());
      assertFalse(option0.hasLongOpt());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.setDescription((String) null);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Option option0 = new Option("", "");
      option0.setArgName("");
      boolean boolean0 = option0.hasArgName();
      assertEquals((-1), option0.getArgs());
      assertFalse(boolean0);
      assertFalse(option0.hasLongOpt());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Option option0 = new Option("", "");
      String string0 = option0.getOpt();
      assertEquals((-1), option0.getArgs());
      assertFalse(option0.hasLongOpt());
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Option option0 = new Option("", true, (String) null);
      option0.setValueSeparator('n');
      option0.addValueForProcessing("");
      assertEquals('n', option0.getValueSeparator());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Option option0 = new Option("", "");
      assertFalse(option0.hasLongOpt());
      
      option0.setLongOpt("");
      String string0 = option0.toString();
      assertEquals("[ option:    ::  ]", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      int int0 = option0.getArgs();
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, false, "The addValue method is not intended for client use. Subclasses should use the addValueForProcessing method instead. ");
      option0.clearValues();
      assertEquals((-1), option0.getArgs());
      assertFalse(option0.hasLongOpt());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, true, (String) null);
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
  public void test18()  throws Throwable  {
      Option option0 = new Option("", true, "");
      boolean boolean0 = option0.hasLongOpt();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Option option0 = new Option((String) null, " ", true, "Han>b?MmLc<LkpTLJrV");
      boolean boolean0 = option0.hasLongOpt();
      assertEquals("Han>b?MmLc<LkpTLJrV", option0.getDescription());
      assertEquals(32, option0.getId());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      assertFalse(option0.hasArgs());
      
      option0.setArgs((-2));
      option0.addValueForProcessing((String) null);
      boolean boolean0 = option0.requiresArg();
      assertEquals((-2), option0.getArgs());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      boolean boolean0 = option0.hasArgName();
      assertEquals((-1), option0.getArgs());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Option option0 = new Option("", "");
      option0.setArgName("z");
      boolean boolean0 = option0.hasArgName();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.setArgs(2847);
      option0.toString();
      assertEquals(2847, option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      assertEquals((-1), option0.getArgs());
      
      option0.setArgs((-2));
      String string0 = option0.toString();
      assertEquals("[ option: null [ARG...] :: null ]", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Option option0 = new Option("", "");
      // Undeclared exception!
      try { 
        option0.addValueForProcessing("");
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
      Option option0 = new Option((String) null, (String) null, false, "The addValue method is not intended for client use. Subclasses should use the addValueForProcessing method instead. ");
      option0.setArgs((-1764));
      option0.setValueSeparator('n');
      // Undeclared exception!
      try { 
        option0.addValueForProcessing("The addValue method is not intended for client use. Subclasses should use the addValueForProcessing method instead. ");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // Cannot add value, list full.
         //
         verifyException("org.apache.commons.cli.Option", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, true, (String) null);
      option0.setValueSeparator('n');
      option0.addValueForProcessing("Illegal option name '");
      assertTrue(option0.hasValueSeparator());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, false, (String) null);
      assertEquals((-1), option0.getArgs());
      
      option0.setArgs(1);
      option0.addValueForProcessing("v~)Y*VT$30)");
      option0.getValue(" :: ");
      assertEquals(1, option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Option option0 = new Option("", "", false, "");
      String string0 = option0.getValue("");
      assertNotNull(string0);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Option option0 = new Option("", true, (String) null);
      option0.addValueForProcessing("");
      try { 
        option0.getValue((-1660));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Option option0 = new Option("", "", true, "");
      String string0 = option0.getValue((-1));
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Option option0 = new Option("", true, (String) null);
      option0.addValueForProcessing("");
      String[] stringArray0 = option0.getValues();
      assertFalse(option0.hasLongOpt());
      assertNotNull(stringArray0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      String[] stringArray0 = option0.getValues();
      assertEquals((-1), option0.getArgs());
      assertNull(stringArray0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Option option0 = new Option("", true, (String) null);
      String string0 = option0.toString();
      assertEquals("[ option:   [ARG] :: null ]", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Option option0 = new Option("", "");
      Class<Object> class0 = Object.class;
      option0.setType(class0);
      String string0 = option0.toString();
      assertEquals("[ option:   ::  :: class java.lang.Object ]", string0);
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Option option0 = new Option("L6", "L6", false, "u");
      Option option1 = (Option)option0.clone();
      boolean boolean0 = option0.equals(option1);
      assertTrue(boolean0);
      assertEquals((-1), option1.getArgs());
      assertNotSame(option1, option0);
      assertEquals(76, option1.getId());
      assertEquals("L6", option1.getLongOpt());
      assertEquals("u", option1.getDescription());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Option option0 = new Option("L6", "L6", false, "u");
      boolean boolean0 = option0.equals(option0);
      assertEquals((-1), option0.getArgs());
      assertTrue(boolean0);
      assertEquals("L6", option0.getLongOpt());
      assertEquals("u", option0.getDescription());
      assertEquals("L6", option0.getOpt());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, true, (String) null);
      boolean boolean0 = option0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Option option0 = new Option("", "");
      boolean boolean0 = option0.equals("");
      assertFalse(boolean0);
      assertFalse(option0.hasLongOpt());
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, true, (String) null);
      Option option1 = new Option("", "HS}saP3uW?Z", true, "");
      boolean boolean0 = option0.equals(option1);
      assertFalse(boolean0);
      assertEquals("", option1.getDescription());
      assertEquals("HS}saP3uW?Z", option1.getLongOpt());
      assertEquals("", option1.getOpt());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      Option option1 = new Option("", (String) null);
      boolean boolean0 = option1.equals(option0);
      assertFalse(boolean0);
      assertEquals((-1), option1.getArgs());
      assertFalse(option1.hasLongOpt());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, true, " ");
      Option option1 = (Option)option0.clone();
      boolean boolean0 = option0.equals(option1);
      assertFalse(option1.hasLongOpt());
      assertTrue(boolean0);
      assertNotSame(option1, option0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Option option0 = new Option("", "", false, "");
      Option option1 = new Option("", true, "");
      boolean boolean0 = option0.equals(option1);
      assertEquals((-1), option0.getArgs());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null, true, (String) null);
      Option option1 = new Option((String) null, "HS}saP3uW?Z", false, "");
      boolean boolean0 = option0.equals(option1);
      assertEquals("HS}saP3uW?Z", option1.getLongOpt());
      assertFalse(boolean0);
      assertEquals("", option1.getDescription());
      assertEquals((-1), option1.getArgs());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Option option0 = new Option("I5", "");
      option0.hashCode();
      assertEquals("I5", option0.getOpt());
      assertEquals("", option0.getDescription());
      assertFalse(option0.hasLongOpt());
      assertEquals((-1), option0.getArgs());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Option option0 = new Option((String) null, true, "");
      option0.setLongOpt("A,;z=;Jy.7h)[FA");
      option0.hashCode();
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Option option0 = new Option("", true, (String) null);
      option0.addValueForProcessing("");
      boolean boolean0 = option0.requiresArg();
      assertFalse(option0.hasLongOpt());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      option0.setOptionalArg(true);
      boolean boolean0 = option0.requiresArg();
      assertTrue(option0.hasOptionalArg());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Option option0 = new Option((String) null, (String) null);
      assertFalse(option0.hasArgs());
      
      option0.setArgs((-2));
      boolean boolean0 = option0.requiresArg();
      assertTrue(boolean0);
  }
}