/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:34:29 GMT 2023
 */

package org.apache.commons.cli2.commandline;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.validation.UrlValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class WriteableCommandLineImpl_ESTest extends WriteableCommandLineImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      UrlValidator urlValidator0 = new UrlValidator();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("", "", 0, 0, 'S', 'P', urlValidator0, "4[p.y@CYTn", linkedList0, 5023);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      String string0 = writeableCommandLineImpl0.getProperty("");
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      PropertyOption propertyOption0 = new PropertyOption("]", "8}UEyoDX_@EzTP\"Yw-", (-514));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getOptionTriggers();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addProperty("Passes properties and values to the application", "P");
      assertFalse(linkedList0.contains("P"));
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      UrlValidator urlValidator0 = new UrlValidator();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("", "", 0, 0, 'S', 'P', urlValidator0, "4[p.y@CYTn", linkedList0, 5023);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      Set set0 = writeableCommandLineImpl0.getProperties();
      assertTrue(set0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption("Switch.already.set", "ay", 2147483645);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      int int0 = writeableCommandLineImpl0.getOptionCount((Option) propertyOption0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      PropertyOption propertyOption0 = new PropertyOption("", "", (-514));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      List list0 = writeableCommandLineImpl0.getNormalised();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, false);
      assertFalse(linkedList0.contains(false));
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      PropertyOption propertyOption0 = new PropertyOption("]", "8}UEyoDX_@EzTP\"Yw-", 1);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addValue(propertyOption0, propertyOption0);
      writeableCommandLineImpl0.addValue(propertyOption0, "]");
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addSwitch(propertyOption0, true);
      // Undeclared exception!
      try { 
        writeableCommandLineImpl0.addSwitch(propertyOption0, false);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Switch already set.
         //
         verifyException("org.apache.commons.cli2.commandline.WriteableCommandLineImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption((Option) propertyOption0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      PropertyOption propertyOption0 = new PropertyOption("]", "8}UEyoDX_@EzTP\"Yw-", 1055);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.addOption(propertyOption0);
      boolean boolean0 = writeableCommandLineImpl0.hasOption((Option) propertyOption0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      UrlValidator urlValidator0 = new UrlValidator();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("", "", 0, 0, 'S', 'P', urlValidator0, "4[p.y@CYTn", linkedList0, 5023);
      LinkedList<Object> linkedList1 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      writeableCommandLineImpl0.addValue(argumentImpl0, argumentImpl0);
      List list0 = writeableCommandLineImpl0.getValues((Option) argumentImpl0, (List) linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(list0, "4[p.y@CYTn", (String) null, 41, 5023);
      List list1 = writeableCommandLineImpl0.getValues((Option) argumentImpl0, (List) linkedList1);
      assertTrue(list1.isEmpty());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      List list0 = writeableCommandLineImpl0.getValues((Option) propertyOption0, (List) null);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "--", "--", (-908), 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      writeableCommandLineImpl0.addSwitch(groupImpl0, true);
      Boolean boolean0 = Boolean.valueOf("&Y]dk9=^c%");
      Boolean boolean1 = writeableCommandLineImpl0.getSwitch((Option) groupImpl0, boolean0);
      assertTrue(boolean1);
      assertNotNull(boolean1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = writeableCommandLineImpl0.getSwitch("$n&");
      assertNull(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      PropertyOption propertyOption0 = new PropertyOption("]", "8}UEyoDX_@EzTP\"Yw-", 1055);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("X2B<IpJro");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      PropertyOption propertyOption0 = new PropertyOption("]", "8}UEyoDX_@EzTP\"Yw-", (-514));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      boolean boolean0 = writeableCommandLineImpl0.looksLikeOption("]");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      UrlValidator urlValidator0 = new UrlValidator();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("", "]Passes properties and values to the application", 280, 280, 'U', '}', urlValidator0, "LshY6*!", linkedList0, 280);
      LinkedList<Object> linkedList1 = new LinkedList<Object>();
      linkedList1.add((Object) "]Passes properties and values to the application");
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList1);
      String string0 = writeableCommandLineImpl0.toString();
      assertEquals("\"]Passes properties and values to the application\"", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      UrlValidator urlValidator0 = new UrlValidator();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("", "", 0, 0, 'U', 'U', urlValidator0, "", linkedList0, 'U');
      LinkedList<Object> linkedList1 = new LinkedList<Object>();
      linkedList1.add((Object) "");
      linkedList1.add((Object) "");
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList1);
      String string0 = writeableCommandLineImpl0.toString();
      assertEquals(" ", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, linkedList0);
      assertEquals("-D", propertyOption0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, (List) null);
      writeableCommandLineImpl0.setDefaultValues(propertyOption0, (List) null);
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Boolean> linkedList0 = new LinkedList<Boolean>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      Boolean boolean0 = new Boolean("Passes properties and values to the application");
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, boolean0);
      assertEquals("-D", propertyOption0.getPreferredName());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<PropertyOption> linkedList0 = new LinkedList<PropertyOption>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      writeableCommandLineImpl0.setDefaultSwitch(propertyOption0, (Boolean) null);
      assertEquals("Passes properties and values to the application", propertyOption0.getDescription());
  }
}