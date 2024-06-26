/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:16:04 GMT 2023
 */

package com.fasterxml.jackson.databind.introspect;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.PropertyName;
import com.fasterxml.jackson.databind.introspect.AnnotatedConstructor;
import com.fasterxml.jackson.databind.introspect.AnnotatedMethod;
import com.fasterxml.jackson.databind.introspect.POJOPropertyBuilder;
import java.util.NoSuchElementException;
import java.util.function.Consumer;
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
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<AnnotatedMethod> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<AnnotatedMethod>((AnnotatedMethod) null, (POJOPropertyBuilder.Linked<AnnotatedMethod>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.MemberIterator<AnnotatedMethod> pOJOPropertyBuilder_MemberIterator0 = new POJOPropertyBuilder.MemberIterator<AnnotatedMethod>(pOJOPropertyBuilder_Linked0);
      boolean boolean0 = pOJOPropertyBuilder_MemberIterator0.hasNext();
      assertFalse(pOJOPropertyBuilder_Linked0.isVisible);
      assertTrue(boolean0);
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked0.isNameExplicit);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
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
  public void test03()  throws Throwable  {
      POJOPropertyBuilder.MemberIterator<AnnotatedConstructor> pOJOPropertyBuilder_MemberIterator0 = new POJOPropertyBuilder.MemberIterator<AnnotatedConstructor>((POJOPropertyBuilder.Linked<AnnotatedConstructor>) null);
      Consumer<AnnotatedConstructor> consumer0 = (Consumer<AnnotatedConstructor>) mock(Consumer.class, new ViolatedAssumptionAnswer());
      pOJOPropertyBuilder_MemberIterator0.forEachRemaining(consumer0);
      assertFalse(pOJOPropertyBuilder_MemberIterator0.hasNext());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("vJu[ve',E");
      POJOPropertyBuilder.Linked<AnnotatedConstructor> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<AnnotatedConstructor>((AnnotatedConstructor) null, (POJOPropertyBuilder.Linked<AnnotatedConstructor>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.MemberIterator<AnnotatedConstructor> pOJOPropertyBuilder_MemberIterator0 = new POJOPropertyBuilder.MemberIterator<AnnotatedConstructor>(pOJOPropertyBuilder_Linked0);
      pOJOPropertyBuilder_MemberIterator0.next();
      assertFalse(pOJOPropertyBuilder_MemberIterator0.hasNext());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
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
  public void test06()  throws Throwable  {
      JsonProperty.Access jsonProperty_Access0 = JsonProperty.Access.READ_WRITE;
      PropertyName propertyName0 = new PropertyName((String) null, (String) null);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<JsonProperty.Access>(jsonProperty_Access0, (POJOPropertyBuilder.Linked<JsonProperty.Access>) null, propertyName0, false, false, false);
      assertFalse(pOJOPropertyBuilder_Linked0.isNameExplicit);
      
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.withoutNext();
      assertFalse(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
      assertFalse(pOJOPropertyBuilder_Linked2.isVisible);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PropertyName propertyName0 = PropertyName.USE_DEFAULT;
      POJOPropertyBuilder.Linked<Integer> pOJOPropertyBuilder_Linked0 = null;
      try {
        pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<Integer>((Integer) null, (POJOPropertyBuilder.Linked<Integer>) null, propertyName0, true, true, true);
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
      PropertyName propertyName0 = new PropertyName("was expecting either valid name character (for unquoted name) or double-quote (for quoted) to start field name");
      POJOPropertyBuilder.Linked<String> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<String>("was expecting either valid name character (for unquoted name) or double-quote (for quoted) to start field name", (POJOPropertyBuilder.Linked<String>) null, propertyName0, true, true, false);
      POJOPropertyBuilder.Linked<String> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withValue("was expecting either valid name character (for unquoted name) or double-quote (for quoted) to start field name");
      assertTrue(pOJOPropertyBuilder_Linked0.isVisible);
      assertTrue(pOJOPropertyBuilder_Linked1.isVisible);
      assertFalse(pOJOPropertyBuilder_Linked1.isMarkedIgnored);
      assertTrue(pOJOPropertyBuilder_Linked1.isNameExplicit);
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      JsonProperty.Access jsonProperty_Access0 = JsonProperty.Access.READ_WRITE;
      PropertyName propertyName0 = new PropertyName((String) null, (String) null);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<JsonProperty.Access>(jsonProperty_Access0, (POJOPropertyBuilder.Linked<JsonProperty.Access>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withoutNext();
      assertSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertFalse(pOJOPropertyBuilder_Linked1.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked1.isVisible);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JsonProperty.Access jsonProperty_Access0 = JsonProperty.Access.READ_WRITE;
      PropertyName propertyName0 = new PropertyName((String) null, "SOLID_MATCH");
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<JsonProperty.Access>(jsonProperty_Access0, (POJOPropertyBuilder.Linked<JsonProperty.Access>) null, propertyName0, true, true, true);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.withoutIgnored();
      assertTrue(pOJOPropertyBuilder_Linked0.isVisible);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertNull(pOJOPropertyBuilder_Linked2);
      assertFalse(pOJOPropertyBuilder_Linked1.isNameExplicit);
      assertTrue(pOJOPropertyBuilder_Linked1.isVisible);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("was expecting either valid name character (for unquoted name) or double-quote (for quoted) to start field name");
      POJOPropertyBuilder.Linked<String> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<String>("was expecting either valid name character (for unquoted name) or double-quote (for quoted) to start field name", (POJOPropertyBuilder.Linked<String>) null, propertyName0, true, true, false);
      POJOPropertyBuilder.Linked<String> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.append(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<String> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.withoutIgnored();
      assertNotNull(pOJOPropertyBuilder_Linked2);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked0);
      assertTrue(pOJOPropertyBuilder_Linked2.isVisible);
      assertFalse(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertTrue(pOJOPropertyBuilder_Linked0.isVisible);
      assertTrue(pOJOPropertyBuilder_Linked2.isNameExplicit);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JsonProperty.Access jsonProperty_Access0 = JsonProperty.Access.READ_WRITE;
      PropertyName propertyName0 = new PropertyName((String) null, "SOLID_MATCH");
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<JsonProperty.Access>(jsonProperty_Access0, (POJOPropertyBuilder.Linked<JsonProperty.Access>) null, propertyName0, true, true, true);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.withoutNonVisible();
      assertSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
      assertTrue(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertNotNull(pOJOPropertyBuilder_Linked2);
      assertTrue(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked2.isNameExplicit);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      JsonProperty.Access jsonProperty_Access0 = JsonProperty.Access.READ_WRITE;
      PropertyName propertyName0 = PropertyName.NO_NAME;
      POJOPropertyBuilder.Linked<Object> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<Object>(jsonProperty_Access0, (POJOPropertyBuilder.Linked<Object>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<Object> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withoutNonVisible();
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertNull(pOJOPropertyBuilder_Linked1);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      JsonProperty.Access jsonProperty_Access0 = JsonProperty.Access.READ_WRITE;
      PropertyName propertyName0 = new PropertyName((String) null, (String) null);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<JsonProperty.Access>(jsonProperty_Access0, (POJOPropertyBuilder.Linked<JsonProperty.Access>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.append(pOJOPropertyBuilder_Linked0);
      assertFalse(pOJOPropertyBuilder_Linked2.isVisible);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
      assertFalse(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      JsonProperty.Access jsonProperty_Access0 = JsonProperty.Access.READ_WRITE;
      PropertyName propertyName0 = new PropertyName((String) null, "SOLID_MATCH");
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<JsonProperty.Access>(jsonProperty_Access0, (POJOPropertyBuilder.Linked<JsonProperty.Access>) null, propertyName0, true, true, true);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.trimByVisibility();
      assertTrue(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertFalse(pOJOPropertyBuilder_Linked2.isNameExplicit);
      assertTrue(pOJOPropertyBuilder_Linked0.isVisible);
      assertTrue(pOJOPropertyBuilder_Linked2.isMarkedIgnored);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked0);
      assertSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
      assertTrue(pOJOPropertyBuilder_Linked2.isVisible);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      JsonProperty.Access jsonProperty_Access0 = JsonProperty.Access.READ_WRITE;
      PropertyName propertyName0 = new PropertyName((String) null, (String) null);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<JsonProperty.Access>(jsonProperty_Access0, (POJOPropertyBuilder.Linked<JsonProperty.Access>) null, propertyName0, false, false, false);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.withNext(pOJOPropertyBuilder_Linked0);
      POJOPropertyBuilder.Linked<JsonProperty.Access> pOJOPropertyBuilder_Linked2 = pOJOPropertyBuilder_Linked1.trimByVisibility();
      assertSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked1);
      assertNotSame(pOJOPropertyBuilder_Linked2, pOJOPropertyBuilder_Linked0);
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PropertyName propertyName0 = new PropertyName("was expecting either valid name character (for unquoted name) or double-quote (for quoted) to start field name");
      POJOPropertyBuilder.Linked<String> pOJOPropertyBuilder_Linked0 = new POJOPropertyBuilder.Linked<String>("was expecting either valid name character (for unquoted name) or double-quote (for quoted) to start field name", (POJOPropertyBuilder.Linked<String>) null, propertyName0, true, true, false);
      POJOPropertyBuilder.Linked<String> pOJOPropertyBuilder_Linked1 = pOJOPropertyBuilder_Linked0.append(pOJOPropertyBuilder_Linked0);
      pOJOPropertyBuilder_Linked1.toString();
      assertFalse(pOJOPropertyBuilder_Linked0.isMarkedIgnored);
      assertTrue(pOJOPropertyBuilder_Linked0.isVisible);
      assertTrue(pOJOPropertyBuilder_Linked1.isNameExplicit);
      assertNotSame(pOJOPropertyBuilder_Linked1, pOJOPropertyBuilder_Linked0);
      assertTrue(pOJOPropertyBuilder_Linked1.isVisible);
      assertFalse(pOJOPropertyBuilder_Linked1.isMarkedIgnored);
  }
}
