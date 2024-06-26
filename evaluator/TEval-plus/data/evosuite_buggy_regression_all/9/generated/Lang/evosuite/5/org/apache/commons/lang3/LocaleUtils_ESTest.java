/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:21:55 GMT 2023
 */

package org.apache.commons.lang3;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import org.apache.commons.lang3.LocaleUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LocaleUtils_ESTest extends LocaleUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      LocaleUtils localeUtils0 = new LocaleUtils();
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Locale locale0 = Locale.GERMAN;
      List<Locale> list0 = LocaleUtils.localeLookupList(locale0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Set<Locale> set0 = LocaleUtils.availableLocaleSet();
      assertEquals(160, set0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      LocaleUtils.SyncAvoid localeUtils_SyncAvoid0 = new LocaleUtils.SyncAvoid();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("et_LU_]>H#");
      assertEquals("LU", locale0.getCountry());
      assertNotNull(locale0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale((String) null);
      assertNull(locale0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: 
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("4bI_*rYj`>o^Ap");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: 4bI_*rYj`>o^Ap
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("r;@b!Yyh7w[");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: r;@b!Yyh7w[
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("uk");
      assertEquals("uk", locale0.toString());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("hl%");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: hl%
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("org.apahe.commons.lang3.LocaleUtils$Syncvoid");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: org.apahe.commons.lang3.LocaleUtils$Syncvoid
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("et__^wU]>5");
      assertEquals("et", locale0.getLanguage());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("cz_tow-Agvq7UsF");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: cz_tow-Agvq7UsF
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("cz_Kow-Agvq7UsF");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: cz_Kow-Agvq7UsF
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Locale locale0 = LocaleUtils.toLocale("ep_RT");
      assertEquals("ep", locale0.getLanguage());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("ep_RT0");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ep_RT0
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      // Undeclared exception!
      try { 
        LocaleUtils.toLocale("ep_RA& ZXpJD2j>$4");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Invalid locale format: ep_RA& ZXpJD2j>$4
         //
         verifyException("org.apache.commons.lang3.LocaleUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      List<Locale> list0 = LocaleUtils.localeLookupList((Locale) null, (Locale) null);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Locale locale0 = new Locale("nmat%ot", "nmat%ot", "nmat%ot");
      List<Locale> list0 = LocaleUtils.localeLookupList(locale0, locale0);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Locale locale0 = Locale.UK;
      List<Locale> list0 = LocaleUtils.localeLookupList(locale0, (Locale) null);
      assertEquals(3, list0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Locale locale0 = new Locale(" w,ewX{)<x4>^J<", "SK", " w,ewX{)<x4>^J<");
      boolean boolean0 = LocaleUtils.isAvailableLocale(locale0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Locale locale0 = Locale.GERMANY;
      boolean boolean0 = LocaleUtils.isAvailableLocale(locale0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      List<Locale> list0 = LocaleUtils.languagesByCountry("TH");
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      List<Locale> list0 = LocaleUtils.languagesByCountry((String) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      List<Locale> list0 = LocaleUtils.countriesByLanguage("th");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      List<Locale> list0 = LocaleUtils.countriesByLanguage((String) null);
      assertEquals(0, list0.size());
  }
}
