/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:04:20 GMT 2023
 */

package org.apache.commons.cli2.option;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Set;
import org.apache.commons.cli2.Argument;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.WriteableCommandLine;
import org.apache.commons.cli2.commandline.WriteableCommandLineImpl;
import org.apache.commons.cli2.option.ArgumentImpl;
import org.apache.commons.cli2.option.Command;
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
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-2089661822), 0);
      Boolean boolean0 = new Boolean("Option.trigger.needs.prefix");
      Switch switch0 = new Switch("cT<08!(k9RC*{m:cT<08!(k9RC*{m:", "Option.trigger.needs.prefix", "Option.trigger.needs.prefix", (Set) null, (String) null, false, (Argument) null, groupImpl0, (-1431655764), boolean0);
      // Undeclared exception!
      try { 
        switch0.validate((WriteableCommandLine) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.cli2.option.ParentImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      NumberValidator numberValidator0 = NumberValidator.getNumberInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("%(#k8fd(6b", "Option.trigger.needs.prefix", 0, 0, 'S', '5', numberValidator0, "%(#k8fd(6b", linkedList0, 0);
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(argumentImpl0, linkedList0);
      argumentImpl0.defaults(writeableCommandLineImpl0);
      assertEquals(0, argumentImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-18), (-18));
      groupImpl0.toString();
      assertEquals(0, groupImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-3), (-3));
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(groupImpl0, linkedList0);
      LinkedList<String> linkedList1 = new LinkedList<String>();
      ListIterator<String> listIterator0 = linkedList1.listIterator();
      boolean boolean0 = groupImpl0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
      assertEquals(0, groupImpl0.getId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      WriteableCommandLineImpl writeableCommandLineImpl0 = new WriteableCommandLineImpl(propertyOption0, linkedList0);
      ListIterator<String> listIterator0 = (ListIterator<String>) mock(ListIterator.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(listIterator0).hasNext();
      doReturn((Object) null).when(listIterator0).next();
      doReturn((Object) null).when(listIterator0).previous();
      boolean boolean0 = propertyOption0.canProcess((WriteableCommandLine) writeableCommandLineImpl0, (ListIterator) listIterator0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-2089661822), 0);
      boolean boolean0 = groupImpl0.equals("Hd{X@tV1l=Wj(:4.");
      assertFalse(boolean0);
      assertEquals(0, groupImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-3), (-3));
      boolean boolean0 = groupImpl0.equals(groupImpl0);
      assertEquals(0, groupImpl0.getId());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-3), (-3));
      PropertyOption propertyOption0 = PropertyOption.INSTANCE;
      boolean boolean0 = groupImpl0.equals(propertyOption0);
      assertEquals(68, propertyOption0.getId());
      assertFalse(boolean0);
      assertEquals(0, groupImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 1175, (-1205));
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, (String) null, "", 0, 0);
      boolean boolean0 = groupImpl0.equals(groupImpl1);
      assertFalse(boolean0);
      assertFalse(groupImpl1.equals((Object)groupImpl0));
      assertEquals(0, groupImpl1.getId());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-18), (-18));
      GroupImpl groupImpl1 = new GroupImpl(linkedList0, "jPno", "jPno", (-18), (-125));
      boolean boolean0 = groupImpl1.equals(groupImpl0);
      assertEquals(0, groupImpl1.getId());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, (-3), (-3));
      groupImpl0.hashCode();
      assertEquals(0, groupImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyOption propertyOption0 = new PropertyOption();
      Option option0 = propertyOption0.findOption("Tu{");
      assertEquals(68, propertyOption0.getId());
      assertFalse(propertyOption0.isRequired());
      assertNull(option0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      NumberValidator numberValidator0 = NumberValidator.getNumberInstance();
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      ArgumentImpl argumentImpl0 = new ArgumentImpl("%(#k8fd(6b", "Option.trigger.needs.prefix", 0, 0, 'S', '5', numberValidator0, "%(#k8fd(6b", linkedList0, 0);
      SourceDestArgument sourceDestArgument0 = new SourceDestArgument(argumentImpl0, argumentImpl0, '*', '/', "S[a0", linkedList0);
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 41, 0);
      LinkedHashSet<DefaultOption> linkedHashSet0 = new LinkedHashSet<DefaultOption>();
      Command command0 = new Command("%(#k8fd(6b", ";a;<N", linkedHashSet0, true, sourceDestArgument0, groupImpl0, 37);
      Option option0 = command0.findOption("%(#k8fd(6b");
      assertEquals(37, option0.getId());
      assertNotNull(option0);
      assertTrue(option0.isRequired());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      LinkedList<Integer> linkedList0 = new LinkedList<Integer>();
      GroupImpl groupImpl0 = new GroupImpl(linkedList0, (String) null, (String) null, 41, 0);
      LinkedHashSet<DefaultOption> linkedHashSet0 = new LinkedHashSet<DefaultOption>();
      groupImpl0.checkPrefixes(linkedHashSet0);
      assertEquals(0, groupImpl0.getId());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedList<Switch> linkedList0 = new LinkedList<Switch>();
      LinkedHashSet<Switch> linkedHashSet0 = new LinkedHashSet<Switch>(linkedList0);
      DefaultOption defaultOption0 = null;
      try {
        defaultOption0 = new DefaultOption("-D-D", "IX}+F]$P8=o", false, "jPno@l2G<O;Ha", (String) null, linkedHashSet0, linkedHashSet0, false, (Argument) null, (Group) null, 461);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Trigger jPno@l2G<O;Ha must be prefixed with a value from java.util.HashSet@0000000003
         //
         verifyException("org.apache.commons.cli2.option.OptionImpl", e);
      }
  }
}