/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:53:41 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.ObjectIdGenerator;
import com.fasterxml.jackson.annotation.SimpleObjectIdResolver;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class POJOPropertyBuilder_ESTest extends POJOPropertyBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder pOJOPropertyBuilder0 = null;
      try {
        pOJOPropertyBuilder0 = new POJOPropertyBuilder((POJOPropertyBuilder) null, propertyName0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      POJOPropertyBuilder.MemberIterator<AnnotatedConstructor> pOJOPropertyBuilder_MemberIterator0 = new POJOPropertyBuilder.MemberIterator<AnnotatedConstructor>((POJOPropertyBuilder.Linked<AnnotatedConstructor>) null);
      // Undeclared exception!
      try { 
        pOJOPropertyBuilder_MemberIterator0.remove();
        fail("Expecting exception: UnsupportedOperationException");
      
      } catch(UnsupportedOperationException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$MemberIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      POJOPropertyBuilder.MemberIterator<AnnotatedConstructor> pOJOPropertyBuilder_MemberIterator0 = new POJOPropertyBuilder.MemberIterator<AnnotatedConstructor>((POJOPropertyBuilder.Linked<AnnotatedConstructor>) null);
      boolean boolean0 = pOJOPropertyBuilder_MemberIterator0.hasNext();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      POJOPropertyBuilder.Linked<AnnotatedConstructor> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<AnnotatedConstructor>((AnnotatedConstructor) null, (POJOPropertyBuilder.Linked<AnnotatedConstructor>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.MemberIterator<AnnotatedConstructor> pOJOPropertyBuilder_MemberIterator0 = new POJOPropertyBuilder.MemberIterator<AnnotatedConstructor>(pOJOPropertyBuilder_Linked0);
      pOJOPropertyBuilder_MemberIterator0.next();
      assertFalse(pOJOPropertyBuilder_MemberIterator0.hasNext());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      POJOPropertyBuilder.MemberIterator<AnnotatedConstructor> pOJOPropertyBuilder_MemberIterator0 = new POJOPropertyBuilder.MemberIterator<AnnotatedConstructor>((POJOPropertyBuilder.Linked<AnnotatedConstructor>) null);
      // Undeclared exception!
      try { 
        pOJOPropertyBuilder_MemberIterator0.next();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$MemberIterator", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("<-=C.orml");
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<PropertyName>(propertyName0, (POJOPropertyBuilder.Linked<PropertyName>) null, propertyName0, false, true, false);
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withoutNonVisible();
      assertFalse(pOJOPropertyBuilder_Linked1.isNameExplicit);
      assertNotNull(pOJOPropertyBuilder_Linked1);
      assertSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertFalse(pOJOPropertyBuilder_Linked1.isMarkedIgnored);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      ObjectIdGenerator<AnnotatedMethod> objectIdGenerator0 = (ObjectIdGenerator<AnnotatedMethod>) mock(ObjectIdGenerator.class, new ViolatedAssumptionAnswer());
      PropertyName propertyName0 = new PropertyName((String) null, "t}<7YZru-G%,O");
      POJOPropertyBuilder.Linked<ObjectIdGenerator<AnnotatedMethod>> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<ObjectIdGenerator<AnnotatedMethod>>(objectIdGenerator0, (POJOPropertyBuilder.Linked<ObjectIdGenerator<AnnotatedMethod>>) null, propertyName0, false, true, true);
      POJOPropertyBuilder.Linked<ObjectIdGenerator<AnnotatedMethod>> pOJOPropertyBuilder_Linked1 = new POJOPropertyBuilder.Linked<ObjectIdGenerator<AnnotatedMethod>>(objectIdGenerator0, pOJOPropertyBuilder_Linked0, pOJOPropertyBuilder_Linked0.name, true, true, true);
      assertTrue(pOJOPropertyBuilder_Linked1.isMarkedIgnored);
      assertTrue(pOJOPropertyBuilder_Linked1.isVisible);
      assertFalse(pOJOPropertyBuilder_Linked1.isNameExplicit);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked0 = null;
      try {
        pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<PropertyName>(propertyName0, (POJOPropertyBuilder.Linked<PropertyName>) null, propertyName0, true, true, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Can not pass true for 'explName' if name is null/empty
         //
         verifyException("com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder$Linked", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      SimpleObjectIdResolver simpleObjectIdResolver0 = new SimpleObjectIdResolver();
      PropertyName propertyName0 = PropertyName.construct("$s+eV*zNmB]*}of");
      POJOPropertyBuilder.Linked<Object> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<Object>(simpleObjectIdResolver0, (POJOPropertyBuilder.Linked<Object>) null, propertyName0, true, true, false);
      assertTrue(pOJOPropertyBuilder_Linked0.isNameExplicit);
      assertTrue(pOJOPropertyBuilder_Linked0.isVisible);
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      LinkedHashMap<PropertyName, String> linkedHashMap0 = new LinkedHashMap<PropertyName, String>();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>(linkedHashMap0, (POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>) null, propertyName0, false, false, false);
      assertFalse(pOJOPropertyBuilder_Linked0.isNameExplicit);
      
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.withoutNext();
      assertFalse(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked2.isVisible);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      LinkedHashMap<PropertyName, String> linkedHashMap0 = new LinkedHashMap<PropertyName, String>();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>(linkedHashMap0, (POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withoutNext();
      assertFalse(pOJOPropertyBuilder_Linked1.isVisible);
      assertFalse(pOJOPropertyBuilder_Linked1.isMarkedIgnored);
      assertSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      LinkedHashMap<PropertyName, String> linkedHashMap0 = new LinkedHashMap<PropertyName, String>();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>(linkedHashMap0, (POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>) null, propertyName0, false, false, false);
      LinkedHashMap<PropertyName, String> linkedHashMap1 = new LinkedHashMap<PropertyName, String>();
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withValue(linkedHashMap1);
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertFalse(pOJOPropertyBuilder_Linked0.isVisible);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      LinkedHashMap<PropertyName, String> linkedHashMap0 = new LinkedHashMap<PropertyName, String>();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>(linkedHashMap0, (POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withValue(linkedHashMap0);
      assertSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertFalse(pOJOPropertyBuilder_Linked1.isVisible);
      assertFalse(pOJOPropertyBuilder_Linked1.isMarkedIgnored);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<PropertyName>((PropertyName) null, (POJOPropertyBuilder.Linked<PropertyName>) null, (PropertyName) null, false, false, true);
      assertFalse(pOJOPropertyBuilder_Linked0.isNameExplicit);
      
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.append(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.withoutIgnored();
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertFalse(pOJOPropertyBuilder_Linked1.isVisible);
      assertNull(pOJOPropertyBuilder_Linked2);
      assertFalse(pOJOPropertyBuilder_Linked0.isVisible);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      LinkedHashMap<PropertyName, String> linkedHashMap0 = new LinkedHashMap<PropertyName, String>();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>(linkedHashMap0, (POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.withoutIgnored();
      assertFalse(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked0.isVisible);
      assertNotNull(pOJOPropertyBuilder_Linked2);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<PropertyName>((PropertyName) null, (POJOPropertyBuilder.Linked<PropertyName>) null, (PropertyName) null, false, false, true);
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.append(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.withoutNonVisible();
      assertTrue(pOJOPropertyBuilder_Linked1.isMarkedIgnored);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertTrue(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertNull(pOJOPropertyBuilder_Linked2);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      LinkedHashMap<PropertyName, String> linkedHashMap0 = new LinkedHashMap<PropertyName, String>();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>(linkedHashMap0, (POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.append(pOJOPropertyBuilder_Linked0);
      assertFalse(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked2.isVisible);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<PropertyName>((PropertyName) null, (POJOPropertyBuilder.Linked<PropertyName>) null, (PropertyName) null, false, false, true);
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.append(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.trimByVisibility();
      assertSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertTrue(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked0.isVisible);
      assertTrue(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("<-=C.orml");
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<PropertyName>(propertyName0, (POJOPropertyBuilder.Linked<PropertyName>) null, propertyName0, false, true, false);
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<PropertyName> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.trimByVisibility();
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked2.isNameExplicit);
      assertTrue(pOJOPropertyBuilder_Linked2.isVisible);
      assertTrue(pOJOPropertyBuilder_Linked0.isVisible);
      assertSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      LinkedHashMap<PropertyName, String> linkedHashMap0 = new LinkedHashMap<PropertyName, String>();
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>(linkedHashMap0, (POJOPropertyBuilder.Linked<HashMap<PropertyName, String>>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<HashMap<PropertyName, String>> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      pOJOPropertyBuilder_Linked1.toString();
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked0.isVisible);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
  }
}