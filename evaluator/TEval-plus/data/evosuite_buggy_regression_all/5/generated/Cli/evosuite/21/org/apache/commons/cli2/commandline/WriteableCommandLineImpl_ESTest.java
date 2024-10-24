/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:27:05 GMT 2023
 */

package org.apache.commons.cli2.commandline;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.FileValidator;
import org.apache.commons.cli2.validation.Validator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WriteableCommandLineImpl_ESTest extends WriteableCommandLineImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      String string0 = writeableCommandLineImpl0.getProperty("A$}g,tBS-?i%KGJUW");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getOptionTriggers();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertEquals(0, set0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      int int0 = writeableCommandLineImpl0.getOptionCount((Option) propertyOption0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getNormalised();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch("-D");
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      propertyOption0.setParent(propertyOption0);
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addOption(propertyOption0);
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, linkedList0);
      linkedList0.add(propertyOption0);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      FileValidator fileValidator0 = new FileValidator();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("-D", "Passes properties and values to the application", (-441), (-441), 'Z', '6', fileValidator0, "Passes properties and values to the application", linkedList0, (-441));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      writeableCommandLineImpl0.addValue(argumentImpl0, propertyOption0);
      linkedList0.add(propertyOption0);
      linkedList0.add(propertyOption0);
      List list0 = writeableCommandLineImpl0.getValues((Option) argumentImpl0, (List) linkedList0);
      assertEquals(2, list0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, propertyOption0);
      writeableCommandLineImpl0.addValue(propertyOption0, linkedList0);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl0.addSwitch(propertyOption0, true);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Switch already set.
         //
         verifyException("org.apache.commons.cli2.commandline.WriteableCommandLineImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch("-D");
      assertNotNull(boolean0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption("Passes properties and values to the application");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addOption(propertyOption0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption((Option) propertyOption0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      linkedList0.add(propertyOption0);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      LinkedList<Properties> linkedList1 = new LinkedList<Properties>();
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, linkedList1);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) linkedList1);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, "-D");
      List list0 = writeableCommandLineImpl0.getUndefaultedValues(propertyOption0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getUndefaultedValues(propertyOption0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty((Option) propertyOption0, "ZY", "-D");
      writeableCommandLineImpl0.addProperty("Switch.no.disabledPrefix", "ZY");
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Command> linkedList0 = new LinkedList<Command>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty("%b=_6=M|Qf{fcrG", "a9p");
      String string0 = writeableCommandLineImpl0.getProperty((Option) propertyOption0, "a9p", "org.apache.commons.cli2.validation.FileValidator");
      assertEquals("org.apache.commons.cli2.validation.FileValidator", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty("Passes properties and values to the application", "Passes properties and values to the application");
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertFalse(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("Passes properties and values to the application");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("-D");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      linkedList0.add("-D");
      linkedList0.add("Passes properties and values to the application");
      String string0 = writeableCommandLineImpl0.toString();
      assertEquals("-D \"Passes properties and values to the application\"", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("n;z]I#:Ipa3aom@a", "n;z]I#:Ipa3aom@a", (-2589), (-2589), 'd', 'h', (Validator) null, "n;z]I#:Ipa3aom@a", linkedList0, 2147483645);
      LinkedList<Integer> linkedList1 = new LinkedList<Integer>();
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, '\u0000', 'd', "n;z]I#:Ipa3aom@a", linkedList1);
      SourceDestArgument sourceDestArgument1 = new SourceDestArgument(sourceDestArgument0, argumentImpl0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(sourceDestArgument1, linkedList1);
      sourceDestArgument1.defaults(writeableCommandLineImpl0);
      assertEquals((-2589), sourceDestArgument1.getMaximum());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = new Boolean("Passes properties and values to the application");
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, boolean0);
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, (Boolean) null);
      assertEquals(68, propertyOption0.getId());
  }
}
