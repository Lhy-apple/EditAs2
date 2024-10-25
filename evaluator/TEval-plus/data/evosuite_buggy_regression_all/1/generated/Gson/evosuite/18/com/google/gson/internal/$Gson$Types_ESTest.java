/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:29:54 GMT 2023
 */

package com.google.gson.internal;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.FilterInputStream;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.WildcardType;
import java.util.Properties;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class $Gson$Types_ESTest extends $Gson$Types_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Type[] typeArray0 = .Gson.Types.getMapKeyAndValueTypes((Type) null, class0);
      assertEquals(2, typeArray0.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Class<Object> class0 = Object.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      Class<?> class1 = .Gson.Types.getRawType(genericArrayType0);
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class1);
      assertNotNull(wildcardType0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      Type type0 = .Gson.Types.resolve(class0, class0, wildcardType0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[1];
      typeArray0[0] = (Type) class0;
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, typeArray0[0], typeArray0);
      Type type0 = .Gson.Types.resolve(typeArray0[0], class0, parameterizedType0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[3];
      typeArray0[0] = (Type) class0;
      typeArray0[1] = (Type) class0;
      typeArray0[2] = (Type) class0;
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(parameterizedType0);
      Properties properties0 = new Properties();
      Object object0 = properties0.put(parameterizedType0, wildcardType0);
      assertNull(object0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      WildcardType wildcardType1 = .Gson.Types.subtypeOf(wildcardType0);
      boolean boolean0 = .Gson.Types.equal(wildcardType1, wildcardType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      WildcardType wildcardType1 = .Gson.Types.supertypeOf(wildcardType0);
      assertTrue(wildcardType1.equals((Object)wildcardType0));
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      Type type0 = .Gson.Types.canonicalize(wildcardType0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      Class<?> class1 = .Gson.Types.getRawType(parameterizedType0);
      assertEquals(1, class1.getModifiers());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      Class<?> class1 = .Gson.Types.getRawType(wildcardType0);
      assertFalse(class1.isEnum());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        .Gson.Types.getRawType((Type) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Expected a Class, ParameterizedType, or GenericArrayType, but <null> is of type null
         //
         verifyException("com.google.gson.internal.$Gson$Types", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      ParameterizedType parameterizedType1 = .Gson.Types.newParameterizedTypeWithOwner(parameterizedType0, parameterizedType0, typeArray0);
      ParameterizedType parameterizedType2 = .Gson.Types.newParameterizedTypeWithOwner(parameterizedType0, parameterizedType0, typeArray0);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType1, (Type) parameterizedType2);
      assertFalse(parameterizedType2.equals((Object)parameterizedType0));
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      boolean boolean0 = .Gson.Types.equal((Object) null, class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      boolean boolean0 = .Gson.Types.equals((Type) class0, (Type) wildcardType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      WildcardType wildcardType1 = .Gson.Types.subtypeOf(class0);
      boolean boolean0 = .Gson.Types.equals((Type) wildcardType0, (Type) wildcardType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType0, (Type) class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      Type[] typeArray1 = new Type[1];
      typeArray1[0] = (Type) parameterizedType0;
      ParameterizedType parameterizedType1 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray1);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType0, (Type) parameterizedType1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(genericArrayType0, genericArrayType0, typeArray0);
      ParameterizedType parameterizedType1 = .Gson.Types.newParameterizedTypeWithOwner(genericArrayType0, parameterizedType0, typeArray0);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType1, (Type) parameterizedType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      boolean boolean0 = .Gson.Types.equals((Type) genericArrayType0, (Type) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      boolean boolean0 = .Gson.Types.equals((Type) null, (Type) genericArrayType0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      boolean boolean0 = .Gson.Types.equals((Type) wildcardType0, (Type) class0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      WildcardType wildcardType1 = .Gson.Types.subtypeOf(class0);
      boolean boolean0 = .Gson.Types.equal(wildcardType1, wildcardType0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      int int0 = .Gson.Types.hashCodeOrZero((Object) null);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      // Undeclared exception!
      try { 
        .Gson.Types.typeToString((Type) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Class<String> class0 = String.class;
      String string0 = .Gson.Types.typeToString(class0);
      assertEquals("java.lang.String", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Class class1 = (Class).Gson.Types.getGenericSupertype(class0, class0, class0);
      assertEquals(1, class1.getModifiers());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Class<String> class0 = String.class;
      Class<Object> class1 = Object.class;
      Class class2 = (Class).Gson.Types.getGenericSupertype(class0, class0, class1);
      assertEquals(1, class2.getModifiers());
      assertNotNull(class2);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      Class<FilterInputStream> class1 = FilterInputStream.class;
      Class class2 = (Class).Gson.Types.getGenericSupertype(class0, class0, class1);
      assertEquals(1, class2.getModifiers());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type type0 = .Gson.Types.getArrayComponentType(class0);
      assertNull(type0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      Type type0 = .Gson.Types.getArrayComponentType(genericArrayType0);
      assertNull(type0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Type[] typeArray0 = .Gson.Types.getMapKeyAndValueTypes(class0, class0);
      assertEquals(2, typeArray0.length);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Class<Object> class0 = Object.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      Class<?> class1 = .Gson.Types.getRawType(genericArrayType0);
      Class class2 = (Class).Gson.Types.resolve(class0, class0, class1);
      assertEquals("class [Ljava.lang.Object;", class2.toString());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf(class0);
      Type type0 = .Gson.Types.resolve(class0, class0, genericArrayType0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.supertypeOf(class0);
      Type type0 = .Gson.Types.resolve(wildcardType0, class0, wildcardType0);
      assertNotNull(type0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Class<Object> class0 = Object.class;
      Type[] typeArray0 = new Type[1];
      // Undeclared exception!
      try { 
        .Gson.Types.newParameterizedTypeWithOwner(typeArray0[0], class0, typeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.gson.internal.$Gson$Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Class<Properties> class0 = Properties.class;
      Type[] typeArray0 = new Type[0];
      ParameterizedType parameterizedType0 = .Gson.Types.newParameterizedTypeWithOwner(class0, class0, typeArray0);
      ParameterizedType parameterizedType1 = .Gson.Types.newParameterizedTypeWithOwner(parameterizedType0, parameterizedType0, typeArray0);
      ParameterizedType parameterizedType2 = .Gson.Types.newParameterizedTypeWithOwner(parameterizedType1, parameterizedType1, typeArray0);
      boolean boolean0 = .Gson.Types.equals((Type) parameterizedType1, (Type) parameterizedType2);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      GenericArrayType genericArrayType0 = .Gson.Types.arrayOf((Type) null);
      boolean boolean0 = .Gson.Types.equal(genericArrayType0, (Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Class<Object> class0 = Object.class;
      WildcardType wildcardType0 = .Gson.Types.subtypeOf(class0);
      boolean boolean0 = .Gson.Types.equal(wildcardType0, class0);
      assertFalse(boolean0);
  }
}
