/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:27:07 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.time.ZoneId;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Set;
import org.apache.commons.cli2.Argument;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.DefaultOption;
import org.apache.commons.cli2.option.GroupImpl;
import org.apache.commons.cli2.option.PropertyOption;
import org.apache.commons.cli2.option.SourceDestArgument;
import org.apache.commons.cli2.option.Switch;
import org.apache.commons.cli2.validation.NumberValidator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class OptionImpl_ESTest extends OptionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LinkedHashSet<DefaultOption> linkedHashSet0 = new LinkedHashSet<DefaultOption>();
      LinkedList<SourceDestArgument> linkedList0 = new LinkedList<SourceDestArgument>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, "mjK", "org.apache.commons.cli2.option.OptionImpl", 777, 777);
      Boolean boolean0 = new Boolean(false);
      Switch switch0 = new Switch("-D", "*[L_Y'BqKW", "d0G)\"{h@Tu@$<Zd0G)\"{h@Tu@$<Z-Dd0G)\"{h@Tu@$<Zd0G)\"{h@Tu@$<Z-D", linkedHashSet0, "Passes properties and values to the application", false, (Argument) null, groupImpl0, 777, boolean0);
      String string0 = switch0.toString();
      assertEquals(777, switch0.getId());
      assertEquals("[-Dd0G)\"{h@Tu@$<Zd0G)\"{h@Tu@$<Z-Dd0G)\"{h@Tu@$<Zd0G)\"{h@Tu@$<Z-D|*[L_Y'BqKWd0G)\"{h@Tu@$<Zd0G)\"{h@Tu@$<Z-Dd0G)\"{h@Tu@$<Zd0G)\"{h@Tu@$<Z-D mjK ()]", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      propertyOption0.defaults((WriteableCommandLine) null);
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      NumberValidator numberValidator0 = NumberValidator.getNumberInstance();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Passes properties and values to the application", "W7tgh0Y.cT_)DKMZ", 1645, 1645, 'y', 'y', numberValidator0, "W7tgh0Y.cT_)DKMZ", linkedList0, 'y');
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, '\u0000', 'y', "Passes properties and values to the application", linkedList0);
      boolean boolean0 = sourceDestArgument0.equals(sourceDestArgument0);
      assertTrue(boolean0);
      assertEquals(121, argumentImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      ListIterator<Integer> listIterator0 = linkedList0.listIterator();
      boolean boolean0 = propertyOption0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
      assertFalse(boolean0);
      assertFalse(propertyOption0.isRequired());
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      ListIterator<GroupImpl> listIterator0 = (ListIterator<GroupImpl>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      propertyOption0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
      assertEquals(68, propertyOption0.getId());
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      boolean boolean0 = propertyOption0.equals("Passes properties and values to the application");
      assertEquals(68, propertyOption0.getId());
      assertFalse(boolean0);
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      PropertyOption propertyOption1 = new PropertyOption((String) null, "-D", (-1719794996));
      boolean boolean0 = propertyOption1.equals(propertyOption0);
      assertEquals((-1719794996), propertyOption1.getId());
      assertFalse(propertyOption1.isRequired());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      NumberValidator numberValidator0 = NumberValidator.getNumberInstance();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("Passes properties and values to the application", "W7tgh0Y.cT_)DKMZ", (-943), (-943), 'y', 't', numberValidator0, "-D", linkedList0, 2326);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, 'T', '1', "W7tgh0Y.cT_)DKMZ", linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-2021640934), (-2021640934));
      boolean boolean0 = sourceDestArgument0.equals(groupImpl0);
      assertEquals(2326, argumentImpl0.getId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption((String) null, "-D", (-1719794996));
      propertyOption0.hashCode();
      assertFalse(propertyOption0.isRequired());
      assertEquals((-1719794996), propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      propertyOption0.hashCode();
      assertEquals(68, propertyOption0.getId());
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption("-D", (String) null, (-1688));
      propertyOption0.hashCode();
      assertEquals((-1688), propertyOption0.getId());
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      Option option0 = propertyOption0.findOption("x~B");
      assertNull(option0);
      assertFalse(propertyOption0.isRequired());
      assertEquals(68, propertyOption0.getId());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      Option option0 = propertyOption0.findOption("-D");
      assertFalse(option0.isRequired());
      assertNotNull(option0);
      assertEquals(68, option0.getId());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedHashSet<PropertyOption> linkedHashSet0 = new LinkedHashSet<PropertyOption>();
      propertyOption0.checkPrefixes(linkedHashSet0);
      assertFalse(propertyOption0.isRequired());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      Set<String> set0 = ZoneId.getAvailableZoneIds();
      // Undeclared exception!
      try { 
        propertyOption0.checkPrefixes(set0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Trigger -D must be prefixed with a value from java.util.HashSet@0000000002
         //
         verifyException("org.apache.commons.cli2.option.OptionImpl", e);
      }
  }
}