/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:54:01 GMT 2023
 */

package org.mockito.internal.util.reflection;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.lang.reflect.WildcardType;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;
import org.mockito.internal.util.reflection.GenericMetadataSupport;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class GenericMetadataSupport_ESTest extends GenericMetadataSupport_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      GenericMetadataSupport.TypeVarBoundedType genericMetadataSupport_TypeVarBoundedType0 = new GenericMetadataSupport.TypeVarBoundedType((TypeVariable) null);
      // Undeclared exception!
      try { 
        GenericMetadataSupport.inferFrom(genericMetadataSupport_TypeVarBoundedType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.util.reflection.GenericMetadataSupport$TypeVarBoundedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      GenericMetadataSupport.TypeVarBoundedType genericMetadataSupport_TypeVarBoundedType0 = new GenericMetadataSupport.TypeVarBoundedType((TypeVariable) null);
      TypeVariable typeVariable0 = genericMetadataSupport_TypeVarBoundedType0.typeVariable();
      assertNull(typeVariable0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      GenericMetadataSupport.TypeVarBoundedType genericMetadataSupport_TypeVarBoundedType0 = new GenericMetadataSupport.TypeVarBoundedType((TypeVariable) null);
      // Undeclared exception!
      try { 
        genericMetadataSupport_TypeVarBoundedType0.interfaceBounds();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.util.reflection.GenericMetadataSupport$TypeVarBoundedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      WildcardType wildcardType0 = mock(WildcardType.class, new ViolatedAssumptionAnswer());
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType0 = new GenericMetadataSupport.WildCardBoundedType(wildcardType0);
      Type[] typeArray0 = genericMetadataSupport_WildCardBoundedType0.interfaceBounds();
      assertEquals(0, typeArray0.length);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      WildcardType wildcardType0 = mock(WildcardType.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(wildcardType0).toString();
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType0 = new GenericMetadataSupport.WildCardBoundedType(wildcardType0);
      WildcardType wildcardType1 = genericMetadataSupport_WildCardBoundedType0.wildCard();
      assertSame(wildcardType1, wildcardType0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      GenericMetadataSupport.TypeVarBoundedType genericMetadataSupport_TypeVarBoundedType0 = new GenericMetadataSupport.TypeVarBoundedType((TypeVariable) null);
      boolean boolean0 = genericMetadataSupport_TypeVarBoundedType0.equals("v%6?i$,K(IFCkmh){8");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      GenericMetadataSupport.TypeVarBoundedType genericMetadataSupport_TypeVarBoundedType0 = new GenericMetadataSupport.TypeVarBoundedType((TypeVariable) null);
      boolean boolean0 = genericMetadataSupport_TypeVarBoundedType0.equals(genericMetadataSupport_TypeVarBoundedType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      GenericMetadataSupport.TypeVarBoundedType genericMetadataSupport_TypeVarBoundedType0 = new GenericMetadataSupport.TypeVarBoundedType((TypeVariable) null);
      boolean boolean0 = genericMetadataSupport_TypeVarBoundedType0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      GenericMetadataSupport.TypeVarBoundedType genericMetadataSupport_TypeVarBoundedType0 = new GenericMetadataSupport.TypeVarBoundedType((TypeVariable) null);
      GenericMetadataSupport.TypeVarBoundedType genericMetadataSupport_TypeVarBoundedType1 = new GenericMetadataSupport.TypeVarBoundedType((TypeVariable) null);
      // Undeclared exception!
      try { 
        genericMetadataSupport_TypeVarBoundedType0.equals(genericMetadataSupport_TypeVarBoundedType1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.util.reflection.GenericMetadataSupport$TypeVarBoundedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Type[] typeArray0 = new Type[0];
      WildcardType wildcardType0 = mock(WildcardType.class, new ViolatedAssumptionAnswer());
      doReturn(typeArray0).when(wildcardType0).getLowerBounds();
      doReturn(typeArray0).when(wildcardType0).getUpperBounds();
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType0 = new GenericMetadataSupport.WildCardBoundedType(wildcardType0);
      // Undeclared exception!
      try { 
        GenericMetadataSupport.inferFrom(genericMetadataSupport_WildCardBoundedType0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.mockito.internal.util.reflection.GenericMetadataSupport$WildCardBoundedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Type[] typeArray0 = new Type[1];
      WildcardType wildcardType0 = mock(WildcardType.class, new ViolatedAssumptionAnswer());
      doReturn((Type[]) null).when(wildcardType0).getLowerBounds();
      doReturn((Type[]) null).when(wildcardType0).getUpperBounds();
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType0 = new GenericMetadataSupport.WildCardBoundedType(wildcardType0);
      // Undeclared exception!
      try { 
        GenericMetadataSupport.inferFrom(genericMetadataSupport_WildCardBoundedType0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.util.reflection.GenericMetadataSupport$WildCardBoundedType", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType0 = new GenericMetadataSupport.WildCardBoundedType((WildcardType) null);
      Object object0 = new Object();
      boolean boolean0 = genericMetadataSupport_WildCardBoundedType0.equals(object0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType0 = new GenericMetadataSupport.WildCardBoundedType((WildcardType) null);
      boolean boolean0 = genericMetadataSupport_WildCardBoundedType0.equals(genericMetadataSupport_WildCardBoundedType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType0 = new GenericMetadataSupport.WildCardBoundedType((WildcardType) null);
      boolean boolean0 = genericMetadataSupport_WildCardBoundedType0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType0 = new GenericMetadataSupport.WildCardBoundedType((WildcardType) null);
      GenericMetadataSupport.WildCardBoundedType genericMetadataSupport_WildCardBoundedType1 = new GenericMetadataSupport.WildCardBoundedType((WildcardType) null);
      // Undeclared exception!
      try { 
        genericMetadataSupport_WildCardBoundedType0.equals(genericMetadataSupport_WildCardBoundedType1);
        fail("Expecting exception: ClassCastException");
      
      } catch(ClassCastException e) {
         //
         // org.mockito.internal.util.reflection.GenericMetadataSupport$WildCardBoundedType cannot be cast to org.mockito.internal.util.reflection.GenericMetadataSupport$TypeVarBoundedType
         //
         verifyException("org.mockito.internal.util.reflection.GenericMetadataSupport$WildCardBoundedType", e);
      }
  }
}